## FCN
### 网络结构

与经典的CNN在卷积层之后使用全连接层得到固定长度的特征向量进行分类（全联接层＋softmax输出）不同，FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的feature map进行上采样, 使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素分类。

### 全卷积网络

* CNN特点
    - 较浅的卷积层感知域较小，学习到一些局部区域的特征
    - 较深的卷积层具有较大的感知域，能够学习到更加抽象一些的特征
    - 高层的抽象特征对物体的大小、位置和方向等敏感性更低，从而有助于识别性能的提高, 所以我们常常可以将卷积层看作是特征提取器
* CNN像素分类很难
    - 存储开销很大：例如对每个像素使用的图像块的大小为15x15，然后不断滑动窗口，每次滑动的窗口给CNN进行判别分类，因此则所需的存储空间根据滑动窗口的次数和大小急剧上升。
    - 计算效率低下：相邻的像素块基本上是重复的，针对每个像素块逐个计算卷积，这种计算也有很大程度上的重复。
    - 像素块大小的限制了感知区域的大小：通常像素块的大小比整幅图像的大小小很多，只能提取一些局部的特征，从而导致分类的性能受到限制。
* 全连接层与全卷积层
    - 对于任一个卷积层，都存在一个能实现和它一样的前向传播函数的全连接层
    - 任何全连接层都可以被转化为卷积层
* FCN输入图像大小任意
    - 对于CNN，一幅输入图片在经过卷积和pooling层时，这些层是不关心图片大小的。对于一个inputsize大小的输入feature map，滑窗卷积，输出outputsize大小的feature map即可。pooling层同理。
    - 但是在进入全连接层时，feature map（假设大小为n×n）要拉成一条向量，而向量中每个元素（共n×n个）作为一个结点都要与下一个层的所有结点（假设4096个）全连接，这里的权值个数是4096×n×n，而我们知道神经网络结构一旦确定，它的权值个数都是固定的，所以这个n不能变化，n是conv5的outputsize，所以层层向回看，每个outputsize都要固定，那每个inputsize都要固定，因此输入图片大小要固定。
* 把全连接层的权重W重塑成卷积层的滤波器有什么好处
    - 在单个向前传播的过程中, 使得卷积网络在一张更大的输入图片上滑动，从而得到多个输出(可以理解为一个label map)

### 反卷积层

* Upsampling的操作可以看成是反卷积(deconvolutional)，卷积运算的参数和CNN的参数一样是在训练FCN模型的过程中通过bp算法学习得到。反卷积层也是卷积层，不关心input大小，滑窗卷积后输出output。deconv并不是真正的deconvolution（卷积的逆变换），最近比较公认的叫法应该是transposed convolution，deconv的前向传播就是conv的反向传播。
* 反卷积参数: 利用卷积过程filter的转置（实际上就是水平和竖直方向上翻转filter）作为计算卷积前的特征图
* 反卷积学习率为0
* 怎么使反卷积的output大小和输入图片大小一致，从而得到pixel level prediction
    - FCN里面全部都是卷积层（pooling也看成卷积），卷积层不关心input的大小，inputsize和outputsize之间存在线性关系。
    - 假设图片输入为[n×n]大小，第一个卷积层输出map就为`conv1_out=(n−kernel)/stride+1`, 记做`conv1_out=f(n)`, 依次类推，`conv5_out=f(conv5_in.size)=f(…f(n))`, 反卷积是要使`n=f′(conv5_out)`成立，要确定`f′`，就需要设置deconvolution层的kernel_size，stride，padding，计算如下：
        ```shell
        layer
        {
            name: "upsample", type: "Deconvolution"
            bottom: "{{bottom_name}}" top: "{{top_name}}"
            convolution_param
            {
                kernel_size: {{2 * factor - factor % 2}}
                stride: {{factor}}
                num_output: {{C}}
                group: {{C}}
                pad: {{ceil((factor - 1) / 2.)}}
                weight_filler: { type: "bilinear" }
                bias_term: false
            }
            param { lr_mult: 0 decay_mult: 0 }
        }
        ```
        factor是指反卷积之前的所有卷积pooling层的累积采样步长，卷积层使feature map变小，是因为stride，卷积操作造成的影响一般通过padding来消除，因此，累积采样步长factor就等于反卷积之前所有层的stride的乘积。

### 跳级结构

* 对CNN的结果做处理，得到了dense prediction，而作者在试验中发现，得到的分割结果比较粗糙，所以考虑加入更多前层的细节信息，也就是把倒数第几层的输出和最后的输出做一个fusion
* 实验表明，这样的分割结果更细致更准确。在逐层fusion的过程中，做到第三行再往下，结果又会变差。具体原因，根据卷积网络的特点，低层网络的信息更接近原图，包含更多噪声；而高层网络的的信息更抽象，包含更多抽象特征。因此，低层的fusion可能导致效果变差。

### 模型训练

* 用AlexNet，VGG16或者GoogleNet训练好的模型做初始化，在这个基础上做fine-tuning，全部都fine-tuning，只需在末尾加上upsampling，参数的学习还是利用CNN本身的反向传播原理
* 采用whole image做训练，不进行patchwise sampling。实验证明直接用全图已经很effective and efficient。项目中之所以进行patchwise sampling，是因为数据量不足，并且原图过大导致显存不足。
* 对class score的卷积层做全零初始化。随机初始化在性能和收敛上没有优势

### 评价

#### 网络逻辑
* 想要精确预测每个像素的分割结果，必须经历从大到小，再从小到大的两个过程
* 在升采样过程中，分阶段增大比一步到位效果更好
* 在升采样的每个阶段，使用降采样对应层的特征进行辅助

#### 缺点
* 得到的结果还是不够精细。进行8倍上采样虽然比32倍的效果好了很多，但是上采样的结果还是比较模糊和平滑，对图像中的细节不敏感
* 对各个像素进行分类，没有充分考虑像素与像素之间的关系。忽略了在通常的基于像素分类的分割方法中使用的空间规整（spatial regularization）步骤，缺乏空间一致性

### Reference

* http://simtalk.cn/2016/11/01/Fully-Convolutional-Networks/
* https://arxiv.org/abs/1411.4038
* http://www.jianshu.com/p/91c5db272725
