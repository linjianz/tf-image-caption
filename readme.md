# tf-image-caption
This is an implementation of the paper [Show and tell: A neural image caption generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf) **(CVPR2015)** with tensorflow. I trained the model under two dataset.

## net flow
![](http://ogmp8tdqb.bkt.clouddn.com//17-10-5/22619444.jpg)  

## MSCOCO (English)
### Program List
- pre-process image and caption  

preprocess_image_mscoco.py  
preprocess_caption_mscoco.py  

- train & test  

train_NIC_mscoco.py  
test_NIC_mscoco.py  

### Result
![](http://ogmp8tdqb.bkt.clouddn.com//17-10-4/86622994.jpg)  
restore from model-100  

1. train set(image_id: 81)  

![](http://ogmp8tdqb.bkt.clouddn.com//17-10-4/23436756.jpg)  

captions | content
:-: | -
gt_1 | a big airplane flying in the big blue sky  
gt_2 | large , two decked , four <UKN> airliner in flight  
gt_3 | an airfrance jet airplane flying in the sky  
gt_4 | a big plane with airfrance on the side of it  
gt_5 | an air france air plane in mid flight  
predicted | a large commercial airplane flying in the sky


2. validation set(image_id: 257)  

![](http://ogmp8tdqb.bkt.clouddn.com//17-10-4/32625703.jpg)  

captions | content
:-: | -
predicted | a group of people walking down a street  

## AI-challenger (Chinese)
### Program List
- pre-process image and caption  

preprocess_image_ai.py  
preprocess_caption_ai.py  

- train & test  

train_NIC_ai.py  
test_NIC_ai.py  

### Result
20 epoches  
![](http://ogmp8tdqb.bkt.clouddn.com//17-10-5/12410664.jpg)  

1. train set

![](http://ogmp8tdqb.bkt.clouddn.com//17-10-5/5489350.jpg)  

captions | content
:-: | -
gt_1 | 球场上 有 两个 穿着 运动服 的 人 在 抢 足球
gt_2 | 宽敞 的 球场上 有 两个 穿着 运动服 的 男人 在 争 足球
gt_3 | 足球场 上 有 两个 穿着 不同 球衣 的 男人 在 抢球
gt_4 | 球场上 有 两个 穿着 运动衣 的 男人 在 抢 足球
gt_5 | 两个 穿着 运动服 的 男人 在 球场上 争抢 足球
predicted | 两个 穿着 球衣 的 男人 在 球场上 争抢 足球

2. validation set

![](http://ogmp8tdqb.bkt.clouddn.com//17-10-5/24886255.jpg)  

captions | content
:-: | -
gt_1 | 一个 穿着 裙子 的 女孩 双手 拿 着 东西 站 在 宽阔 的 草地 上
gt_2 | 宽阔 的 草地 上 站 着 一个 双手 拿 着 果子 的 孩子
gt_3 | 草地 上 一个 披 着 长发 的 女孩 在 亲 水果
gt_4 | 茂盛 的 草地 上 有 一个 穿着 白色 的 连衣裙 的 女孩 在 亲吻 水果
gt_5 | 绿油油 的 草地 上 站 着 一个 双手 拿 着 水果 的 女孩
predicted | 一个 双手 拿 着 花 的 女人 站 在 茂盛 的 草丛里

