import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.imshow('check_imgs/img.png', dataset[0][0])
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        #print(pil_image.size)
        pil_image = pil_image.resize((pil_image.size[0]//2,pil_image.size[1]//2), Image.ANTIALIAS)
        #print(pil_image.size)

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


'''
class slDataset(Dataset):
    def __init__(self,transform=transform,bKP=False,bFDNA=False,blockA=False,train=0,giveIndex=False):
        self.transform = transform
        self.blockA = blockA##for speed do not load A images -- relies on precalced fdna
        self.giveIndex=giveIndex#give back also filename from iter

        self.M = []   #d[0]##imagename is [18:]
        self.A = []# d[2]##imagename ends in "_1.jpg"

        if True:#older data from Julia
            d = pickle.load(open(path + fname, 'rb'))
            print ("old dump SL files",len(d[0]))
            for namem,setA in zip(d[0],d[2]):
                self.M.append((p1a, namem[18:]))
                self.A.append((p2a, setA))#part of path and lists of articles
        if use2019data:#newer data dump from DataLake, july 2019
            d = pickle.load(open(path+"outfits_uniqueJuly2019.dat", 'rb'))
            print ("new dump SL files", len(d[0]))
            for namem,setA in zip(d[0],d[2]):
                self.M.append((p1b, namem[18:]))
                self.A.append((p2b,setA))

        if use2021:
            f21 = os.listdir(path2021+"m")
            f21.sort()
            relPath = "../fullbody_models_Jan2021/"
            for namem in f21:
                self.M.append((relPath+"m/",namem))
                csku=namem[:-7]
                self.A.append((relPath+"a/", [csku]))
            print ("new added files"+str(len(f21)))

        filter_article_silhouettes(self.A)
        #filterArticleSets(self.A)

        if opt.SMPL:
            smp = "/home/njetchev/opensource/SPIN/"
            dat0= pickle.load(open(smp+'SL_smpl0.dat','rb'))
            dat1= pickle.load(open(smp+'SL_smpl1.dat','rb'))
            print ("data",len(dat0),len(dat1),list(dat0.keys())[0],list(dat1.keys())[0])
            self.smpl = []
            for p, n in self.M:
                name = path + p + n
                if name in dat0:
                    v = dat0[name]
                elif name in dat1:
                    v = dat1[name]
                else:
                    v=np.zeros((1,157))
                    #raise Exception("missing model smlpl params"+name)
                self.smpl.append(torch.FloatTensor(v))
            self.smpl=torch.cat(self.smpl).half()
            print ("smpl vecs prepared",self.smpl.shape)

        cardinality = set()
        for p,setA in self.A:
            cardinality.add(len(setA))
        print ("MA",len(self.M),len(self.A),type(self.M),"cardinality A per M",len(cardinality))
        self.KP = bKP#use keypints of model as well

        self.flatKP={}
        if self.KP:##init vectors
            import json

            if True:#for old data
                with open(kpath + kfile2a, 'r') as f:
                    kpfiles = json.load(f)
                a = np.load(kpath + kfile1a)
                for i in range(len(kpfiles)):
                    name = kpfiles[i]
                    ix = 55  # name.rfind('/')
                    name = name[ix:]
                    self.flatKP[name] = rescaleKP(a[i])

            #for new data
            if use2019data:
                with open(kpath + kfile2b, 'r') as f:
                    kpfiles = json.load(f)
                a = np.load(kpath + kfile1b)
                for i in range(len(kpfiles)):
                    name = kpfiles[i]
                    ix = 60  # name.rfind('/')
                    name = name[ix:]
                    self.flatKP[name] = rescaleKP(a[i])

            if use2021:
                kpfiles,a = pickle.load(open(path2021+'fullbodyposes.dat','rb'))
                print ("new kps",len(kpfiles))
                for i in range(len(kpfiles)):
                    self.flatKP[kpfiles[i]] = rescaleKP(a[i])

            print ("flat KP pose inited",len(self.flatKP))
            ##now make into a big array, with same indexorder as M
            flatKP=[]
            miss=0

            for p1, name in self.M:
                if name in self.flatKP:
                    flatKP.append(self.flatKP[name].unsqueeze(0))##1,16,3
                else:
                    flatKP.append(torch.zeros(1,16,3))
                    miss +=1
                    #print ("miss",p1,len(name))##usually poorly parsed from DataLake -- no image, iter will throw exceptio anywaay

            print ("total KP",len(flatKP),"miss",miss)
            self.flatKP = torch.cat(flatKP)#B,16,3
            print ("KP array",self.flatKP.shape)

        if useVideo:
            #1. add A and M files
            #2. add KPoses
            #3. -- no need to do for FDNA anything, linked via csku name
            avdat = pickle.load(open(path+"vidarticles.dat",'rb'))##just one of two folders where A image is located
            vpath= path+"filteredVFrames/"
            svfiles = set(os.listdir(vpath))#set of all filtered frame filenames

            vkp_path = "/home/teamshare/Zalando_Research/style_interpolation/video_keypoints/"
            kpbuf = []  ##for frame skeletons

            print ("last M,A",self.M[-1],self.A[-1])
            for posef in os.listdir(vkp_path):##store all pose vectors for later use
                try:
                    poses = np.load(vkp_path + posef)
                except Exception as e:
                    print ("kpv error",e)
                    continue
                csku=posef[:-4]#remove .npy suffix
                for t in range(poses.shape[0]):
                    frame= csku+"_f%03d.jpg"%(t)
                    if frame in svfiles and csku in avdat:#so article image A present, also pose present and filtered good
                        self.M.append(("filteredVFrames/",frame))
                        self.A.append((avdat[csku],[csku]))#setA of 1 element
                        kpbuf.append(rescaleKP(poses[t]).unsqueeze(0))#1 16 3
            kpbuf=torch.cat(kpbuf,0)
            print ("all new video frame poses",kpbuf.shape)
            self.flatKP = torch.cat([self.flatKP,kpbuf],0)##poses of all inside
            print ("flatKP after concat",self.flatKP.shape)
            print ("A and M",len(self.A),len(self.M))
            print ("last M,A", self.M[-1], self.A[-1])

        self.FDNA = bFDNA
        if bFDNA and not opt.vgg_spatial:
            self.FDNA = bFDNA
            cfg,afdna=pickle.load(open(path+"amortize_fdna2019.dat", 'rb'))
           
            self.fdna= {}
            for p,setA in self.A:
                for aa in setA:##article names
                    self.fdna[aa]=None
            countArticles = len(self.fdna)
            print ("total articles",countArticles)

            vggskip = 0

            if bVGG:
                dicVGG = pickle.load(open(path+"DAE_articles2.dat", 'rb'))#shoplook_util/ #from amortize texture descriptor

            brnan=0
            for i in range(len(cfg)):
                name=cfg[i]##my dump uses directly strings
                if name in self.fdna:
                    embed = torch.FloatTensor(np.float32(afdna[i]))
                    #add here VGG features in case second array available
                    if bVGG:
                        if name in dicVGG:
                            vembed=dicVGG[name]
                        else:
                            vembed=np.zeros(nEmbed-128)
                            vggskip+=1
                        embed = torch.cat([embed,torch.FloatTensor(vembed)]).half()
                    else:
                        if nEmbed > 128 and bVGG:
                            vembed = np.zeros(nEmbed - 128)
                            embed = torch.cat([embed, torch.FloatTensor(vembed)]).half()

                    if embed.float().sum()!=embed.float().sum():
                        print ("nan error")
                        embed=torch.zeros(nEmbed).half()
                        brnan+=1
                    self.fdna[name]=embed
                    #if np.isnan(self.fdna[name].sum()):
                    #    brnan+=1
            print ("brnan miss fdna", brnan)
            print ("vgg miss",vggskip)

            if use2021:
                cskus,embeddings = pickle.load(open(path2021+'embed640_2021.dat', 'rb'))
                nnan =0
                for i in range(len(cskus)):
                    name = cskus[i]
                    if np.isnan(embeddings[i].sum()):
                        nnan+=1
                    else:
                        self.fdna[name] = torch.FloatTensor(embeddings[i])
                print ("total %d nans %d 2021 Aembed"%(len(cskus),nnan))

            for name in list(self.fdna.keys()):
                if self.fdna[name] is None or self.fdna[name].float().sum()==0:#np.isnan(self.fdna[name].sum()):
                    del self.fdna[name]
                    #print ("missing article",name)
            print ("total fdna vectors",len(self.fdna)," from articles",countArticles)

            self.fdna_names={}#string name mapped to index in fdna array
            fdna=[]##array to store all
            br=0
            for name in list(self.fdna.keys()):
                self.fdna_names[name]=br
                fdna.append(self.fdna[name].unsqueeze(0))
                br+=1
            self.fdna= torch.cat(fdna)
            print ("fdna array",self.fdna.shape)

        if opt.VQGAN:
            vqgan,vqnames =torch.load('indexData_name.dat')
            self.vqgan={}
            for i in range(len(vqnames)):
                self.vqgan[vqnames[i]]=vqgan[i]            
            print ("vqdata prepared",len(self.vqgan))

        if train >0:
            N = len(self.M)
            sp = N//100*85
            if train ==1:
                self.M=self.M[:sp]
                self.A = self.A[:sp]
                self.flatKP=self.flatKP[:sp]

            if train==2:
                self.M = self.M[sp:]
                self.A = self.A[sp:]
                self.flatKP = self.flatKP[sp:]
            print ("len after traintest split",len(self.M))

    def getX(self, name, type="M",skv=None):
        if type == "M":
            name = path+name# p1+name[18:]
        elif type == "A":
            name = path+name+"_1.jpg"#+p2
        else:
            name = path + skelSavePath + "skel_"+name#18:
        im = Image.open(name)
        if im.mode != "RGB":#some articles bugged
            im=im.convert("RGB")
        if im.size != (762, 1100):
            im = im.resize((762, 1100), resample=Image.LANCZOS)  ##hack for 1 buggy image in data

        ##new Nov. 18 align w.r.t. keypoints
        if skv is not None and bAlignM:
            mean = skv[:,1].mean()
            delta= mean#if positive needs to move image so much to the right; in -1 to 1 coords
            deltaw=169-int(delta*0.5*1100)#in pixels
            deltaw=max(deltaw,1)
            deltaw = min(deltaw, 2*169-1)
            #print (mean,"deltaw",deltaw,169*2-deltaw)
            im= transforms.Pad((deltaw,0,169*2-deltaw,0),padding_mode='edge')(im)
            #print ("im size",im.size)

        if self.transform is not None:
            im = self.transform(im)[:3, :, :]
        return im

    def __len__(self):
        return len(self.M)

    ##issue: how to give back element of set -- will dataLoader use a list for this??
    def __getitem__(self, index,verbose=False):
        try:
            #assert(index not in badPose5)
            p1,namem = self.M[index]

            if p1[:6]=='filter':##chance to reduce video data rate
                if random.randrange(10) <5 and not self.giveIndex:# chance to throw again and get normal item; if analyzing do not do
                    raise Exception

            if verbose:
                print (p1,namem,index)

            if self.KP:
                skv = 1.0*self.flatKP[index]
                if not filterSkelet(skv):
                    if verbose:
                        print ('poor skeleton',skv.shape,skv)
                    return self.__getitem__(random.randrange(self.__len__()))
            else:
                skv=None
            if verbose:
                print ("good skeleton",p1,namem)
            M = self.getX(p1+namem,skv=skv)

            A=[]
            p2,setA=self.A[index]

            random.shuffle(setA)##so that meaningful trim possible in training
            #  note that this changes the data structure -- consistent when saving and analyzing; but then use workers=0

            if not len(setA)>0:
                if verbose:
                    print ("empty articles")
                return self.__getitem__(random.randrange(self.__len__()))

            for csku in setA:
                if not self.blockA:
                    A.append(self.getX(p2+csku,"A"))
                else:
                    A.append(M*1)#speed hack when training, still non-zero variance
            while (len(A))<7:
                A.append(A[-1]*0)

            if opt.IRect > 1:
                M = M[:,:,M.shape[2]//4:-M.shape[2]//4]##make rectangular

            if not self.KP:
                return M,A
            else:
                if bAlignM:
                    skv[:,1]-=skv[:,1].mean()##careful with inplace change -- that is why we copy skv above
                #assert (gmm.score(align_skv(skv).view(1,48)) > 68)
                #if verbose:
                #    print ("passed GMM")

                if not self.FDNA:
                    if self.giveIndex:
                        return M, A, skv,(index,setA)

                    if opt.VQGAN:
                        return M,A,skv,self.vqgan[(p1,namem)].unsqueeze(0)

                    return M,A,skv
                else:
                    fdna = []##embedding for each article
                    if bVGG_spatial:
                        vgg_spatial =[]
                    for csku in setA:
                        try:
                            if bVGG_spatial:
                                #vgg88 = torch.FloatTensor(pickle.load(open("shoplook_util/VGG8x8/%s.dat" % (csku), 'rb')))#512x8x8
                                #embed=torch.cat([embed,vgg88.mean(2).mean(1)])
                                nameE = "/home/teamshare/Zalando_Research/shoplook_2017/DAEfeatures/"+csku+".npy"
                                xE=torch.FloatTensor(np.load(nameE))
                                embed=xE[:640].mean(2).mean(1)###FDNA predict and mean of other features
                                #if opt.vgg_spatial_size <8:
                                #    vgg88=F.interpolate(vgg88.unsqueeze(0),size=(opt.vgg_spatial_size,opt.vgg_spatial_size),mode='bilinear')[0]
                                #efdna=self.fdna[ix].unsqueeze(2).unsqueeze(3).expand(-1,-1,opt.vgg_spatial_size,opt.vgg_spatial_size)
                                #vgg_spatial.append(torch.cat([efdna,vgg88]),1)#(512+128)x8x8
                                vgg_spatial.append(xE[128:640])
                            else:#normal fdna case
                                ix = self.fdna_names[csku]
                                embed = self.fdna[ix].float() * 1.0
                            fdna.append(embed)#name
                            if verbose:
                                print (csku,"article embed",embed.shape)
                        except Exception as e:
                            if verbose:
                                print ("vgg error",e)
                            fdna.append(torch.zeros(nEmbed-(1-int(bVGG))*(nEmbed-128)))
                            #HACK for nan's in FDNA data, also support the fullA case with bVGG=False -- only 128 then
                            if bVGG_spatial:
                                vgg_spatial.append(torch.zeros(512+128*0,opt.vgg_spatial_size,opt.vgg_spatial_size))

                    while len(fdna)<7:#complete to 7 per item
                        A[len(fdna)]*=0##so we know no proper article, trim lists
                        fdna.append(torch.zeros(nEmbed))
                        if bVGG_spatial:
                            vgg_spatial.append(torch.zeros(512,opt.vgg_spatial_size,opt.vgg_spatial_size))

                    if bVGG_spatial:
                        assert(len(fdna)==len(vgg_spatial))
                    #new logic Aug 30 2021, for new puppet  features
                    #skel=  []  #torch.zeros(1)##if ncSkelet==16
                    #for csku in setA:
                    #    skel.append(csku)
                    #while len(skel)< 7:
                    #    skel.append('no_csku')      
                    cskus=       np.array(setA).reshape(1,-1)#so dummy batch with 1 instances       
                    #print ('cskus',cskus,cskus.shape)
                    pupath="/home/teamshare/Zalando_Research/fullbody_models_Jan2021/puppets/"
                    skel= Image.open('%spuppet%d.jpg'%(pupath,index))#excception if empty
                    dep= Image.open('%sdpuppet%d.jpg'%(pupath,index))
                    skel = torch.cat([puppet_transform(skel),puppet_transform(dep)[:1]])##C=3+1 x 1024 x512
                    #get_mesh_depth(M.unsqueeze(0)*0.5+0.5,self.modelSMPL,cskus=cskus,initedMGN=self.initedMGN,laplaceCache=self.laplaceCache).squeeze()#so 3xHxW
                    
                    if verbose:
                        print ("good file overall", p1, namem)

                    if opt.FaceD:
                        if not bVGG_spatial:
                            return M, A, skv, fdna, skel,getFaceD(namem)
                        else:
                            return M, A, skv, fdna, skel,getFaceD(namem),vgg_spatial
                    elif bVGG_spatial:
                        return M, A, skv, fdna, skel,vgg_spatial
                    else:
                        assert(len(A) >0)#weird, should not come here...
                        #print ("output",M.shape,len(A),skv.shape,len(fdna),"3dpuppet",skel.shape)
                        return M, A, skv, fdna,skel##main case actually
                    return M, A, skv,fdna
        except Exception as e:
            if verbose:
                print ("itere",e) ##usually file name too long error
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                raise Exception
            return self.__getitem__(random.randrange(self.__len__()))
'''
