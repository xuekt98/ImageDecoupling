# Image Decoupling 图像解耦实验
## Image Recolor Model Version3
### VQGAN第一部分 Codebook的训练
利用Lab图像构建codebook，然后通过codebook序列重建出ab图像  
网络结构为VQGAN的第一部分网络结构  
损失函数为ab损失  

### VQGAN的第二部分 GPT模型的训练
相对于Version1，用VGG16将L分量的图像变换为8x8x1024维的图像，作为transformer的输入。  
网络结构：L分量通过VGG16变换为8x8x1024维向量，之后与VQGAN中GPT部分网络结构一样  
损失函数删除了gumbel_softmax部分，只保留了交叉熵损失，

### 存在的问题
仍然存在不能精准的识别图像中物体的边缘的问题，但是比Version1模型和Version2模型效果要好  

### ImageNet ILSVRC2012的结果
当前模型在ImageNet ILSVRC2012数据集上的结果：在一些图像上的着色效果很好，而且可以产生  
不同风格的图像。但是模型存在一个重大的缺陷，在GPT进行预测序列的时候，序列最开始预测到某些  
值，比如在此次实验中序列号131，之后预测的结果将会全是131，导致最终染不上色的情况。目前还  
没有找到解决的方法。  
实验证明本模型在自动着色方面确实可以做到精确到每一个像素，这是本次实验一个重要的收获。
