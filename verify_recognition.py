import torchvision
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np


from siamese_utils import Normalize, DDFA_Pairs_Dataset
from verification import calculate_roc, generate_roc_curve

import matplotlib.pylab as plt
from siamese_network import load_resnet50

def sia_extract_feature_ddfa(checkpoint_fp, root, pairs_txt, log_dir, device_ids = [0],
                      batch_size = 32, num_workers = 8):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']     ## 把张量从GPU 0~7 移动到 GPU 0, get paramerm's weight
    torch.cuda.set_device(device_ids[0])                                                #bing cong zi dian zhong na chu key=='state_dict' de nei rong
    model = load_resnet50()    #get a model explain or document
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint)
    dataset = DDFA_Pairs_Dataset(root, pairs_txt,
                              transform=transforms.Compose([transforms.ToTensor(), Normalize(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cudnn.benchmark = True   #总的来说，大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    model.eval()   #jiang model bian cheng test pattern


    embeddings_l = []   # metrix: len(dataset)xmodel.FEATURE_DIM  === 6000x512
    embeddings_r = []
    pairs_match = []
    with torch.no_grad():     #bu ji suan gradient
        for i, (inputs_l, inputs_r, matches) in enumerate(data_loader):

            inputs_l = inputs_l.cuda()           #fang dao gpu shang jin xing yun suan 
            inputs_r = inputs_r.cuda()           #fang dao gpu shang jin xing yun suan 
            feature_l, feature_r = model(inputs_l, inputs_r)

            feature_l = feature_l.div(torch.norm(feature_l, p=2, dim=1, keepdim=True).expand_as(feature_l))
            feature_r = feature_r.div(torch.norm(feature_r, p=2, dim=1, keepdim=True).expand_as(feature_r))

            for j in range(feature_l.shape[0]): #output.shape[0] = 128 == batch_size
                feature_l_np = feature_l[j].cpu().numpy().flatten()
                feature_r_np = feature_r[j].cpu().numpy().flatten()
                matches_np = matches[j].cpu().numpy().flatten()

                embeddings_l.append(feature_l_np)
                embeddings_r.append(feature_r_np)
                pairs_match.append(matches_np)

    embeddings_l = np.array(embeddings_l)
    embeddings_r = np.array(embeddings_r)
    pairs_match = np.array(pairs_match)

    pairs_match = pairs_match.reshape(6000)    #6000 pairs images
    thresholds = np.arange(0, 4, 0.0001)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings_l, embeddings_r, pairs_match, nrof_folds = 10, pca = 0)
    
    generate_roc_curve(fpr, tpr, log_dir)

    return tpr, fpr, accuracy, best_thresholds, embeddings_l, embeddings_r

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
         bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

if __name__ == '__main__':
    
#    checkpoint_fp = "training_debug/logs/wpdc+shp+identification_1/_checkpoint_epoch_50.pth.tar"
#    root_lfw = "/home/luoyao/Project_3d/compare-test/facenet-master/datasets/align_lfw"
#    pairs_txt = "/home/luoyao/Project_3d/compare-test/facenet-master/data/pairs.txt"
#    log_dir = "/home/luoyao/Project_3d/3D_face_solution/Siamese_network/recongnition"
#
#    tpr, fpr, accuracy, best_thresholds, dist, pairs_match, embeddings_l = sia_extract_feature_lfw(checkpoint_fp, root_lfw, pairs_txt, log_dir)

    ############# DDFA #####################
    checkpoint_fp = "training_debug/logs/model/_checkpoint_epoch_50.pth.tar"
    root_ddfa = "/home/luoyao/Project_3d/3D_face_solution/3DDFA_TPAMI/3DDFA_PAMI/train_aug_120x120" 
    pairs_txt = "/home/luoyao/Project_3d/3D_face_solution/Siamese_network/recongnition/pairs_ddfa.txt"
    log_dir = "training_debug/logs/contrastive_loss/recognition/result.png"

    tpr, fpr, accuracy, best_thresholds, embeddings_l, embeddings_r = sia_extract_feature_ddfa(checkpoint_fp, root_ddfa, pairs_txt, log_dir)
    mean_acc = np.mean(accuracy)
    print(mean_acc)

#    tpr_, fpr_, acc = calculate_accuracy(0.15, dist, pairs_match)


    ## abserve batch pairs-lfw data
#    dataset = LFW_Pairs_Dataset(root_lfw, pairs_txt,
#                              transform=transforms.Compose([transforms.ToTensor()]))
#    
#    train_loader = data.DataLoader(dataset, batch_size = 8, num_workers=8,
#                              shuffle=False, pin_memory=True, drop_last=True)
#    dataiter = iter(train_loader)
#    
#    example_batch = next(dataiter)
#    concatenated = torch.cat((example_batch[0],example_batch[1]), 0)
#    imshow(torchvision.utils.make_grid(concatenated))
#    print(example_batch[2])























