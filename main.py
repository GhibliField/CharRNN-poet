#coding:utf8
import sys,os
import torch as t
from data import get_data
from model import PoetryModel
from torch import nn
from torch.autograd import Variable
from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb

class Config(object):
    data_path = 'data/' # 诗歌的文本文件存放路径
    pickle_path= 'tang.npz' # 预处理好的二进制文件 
    author = None # 只学习某位作者的诗歌
    constrain = None # 长度限制
    category = 'poet.tang' # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3 
    weight_decay = 1e-4
    use_gpu = True
    epoch = 20  
    batch_size = 128
    maxlen = 125 # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 20 # 每20个batch 可视化一次
    # use_env = True # 是否使用visodm
    env='poetry' # visdom env
    max_gen_len = 200 # 生成诗歌最长长度
    debug_file='/tmp/debugp'
    model_path=None # 预训练模型路径
    prefix_words = '细雨鱼儿出,微风燕子斜。' # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words='闲云潭影日悠悠' # 诗歌开始
    acrostic = False # 是否是藏头诗
    model_prefix = 'checkpoints/tang' # 模型保存路径

opt = Config()

def generate(model,start_words,ix2word,word2ix,prefix_words=None):
    '''
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：

    '''
    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = Variable(t.Tensor([word2ix['<START>']]).view(1,1).long())
    if opt.use_gpu:input=input.cuda()
    hidden = None

    if prefix_words:#用以控制诗歌的意境与长短
        for word in prefix_words:
            output,hidden = model(input,hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1,1)
    
    for i in range(opt.max_gen_len):
        output,hidden = model(input,hidden)
  
        if i<start_word_len:#将连续的好几个字作为开头串依次作为输入，计算隐藏元
            w = results[i]
            input = Variable(input.data.new([word2ix[w]])).view(1,1)      
        else:
            top_index  = output.data[0].topk(1)[1][0]
            w = ix2word[top_index]  
            results.append(w)
            #用预测的词作为新的输入，计算隐藏元和预测新的输出
            input = Variable(input.data.new([top_index])).view(1,1)
        if w=='<EOP>':
            del results[-1]#del 删除的是变量对数据的引用，而非数据
            break     
    return results

def gen_acrostic(model,start_words,ix2word,word2ix, prefix_words = None):
    '''
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    '''
    results = []
    start_word_len = len(start_words)
    input = Variable(t.Tensor([word2ix['<START>']]).view(1,1).long())
    if opt.use_gpu:input=input.cuda()
    hidden = None
    
    index=0 # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word='<START>'

    if prefix_words:
        for word in prefix_words:
            output,hidden = model(input,hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1,1)

    for i in range(opt.max_gen_len):
        output,hidden = model(input,hidden)
        top_index  = output.data[0].topk(1)[1][0]
        w = ix2word[top_index]

        if (pre_word  in {u'。',u'！','<START>'} ):
            # 如果遇到句号，藏头的词送进去生成

            if index==start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:  
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index+=1
                input = Variable(input.data.new([word2ix[w]])).view(1,1)    
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = Variable(input.data.new([word2ix[w]])).view(1,1)
        results.append(w)
        pre_word = w
    return results


def train(**kwargs):

    for k,v in kwargs.items():
        setattr(opt,k,v)

    vis = Visualizer(env=opt.env)

    # 获取数据
    data,word2ix,ix2word = get_data(opt)
    data = t.from_numpy(data)#把数据类型转为tensor
    dataloader = t.utils.data.DataLoader(data,#初始化Dataloader类实例
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256)#(vocab_size, embedding_dim, hidden_dim)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()#损失函数定义为交叉熵
    
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()
    loss_meter = meter.AverageValueMeter()

    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii,data_ in tqdm.tqdm(enumerate(dataloader)):    #tqdm进度条工具
            #取一个batch的数据
            # 训练

            #data_.size:(batch_size,maxlen)
            data_ = data_.long().transpose(1,0).contiguous()#转置后返回一个内存连续的有相同数据的tensor
            # if epoch==0 and ii ==0:
            #     print('size of data_ after transpose: \n',data_.size())
            if opt.use_gpu: data_ = data_.cuda()      
            optimizer.zero_grad()#梯度清零

            input_,target = Variable(data_[:-1,:]),Variable(data_[1:,:])#input_是所有句子的前maxlen-1个item的集合，
            #target是所有句子的后maxlen-１个item的集合
            #以＂床前明月光＂为例，输入是＂床前明月＂，要预测＂前明月光＂
            output,_  = model(input_)
            #Tensor.view(-1)按照第０个维度逐个元素读取将张量展开成数组

            loss = criterion(output,target.view(-1))
            loss.backward()
            optimizer.step()
        
            loss_meter.add(loss.data[0])

            # 可视化
            if (1+ii)%opt.plot_every==0:

                if os.path.exists(opt.debug_file):#如果存在调试文件，
                    #则进入调试模式
                    ipdb.set_trace()

                vis.plot('loss',loss_meter.value()[0])
                
                # 诗歌原文
                poetrys=[ [ix2word[_word] for _word in data_[:,_iii]] #每一个句子（诗歌）的每一个item（id）要转换成文本
                                    for _iii in range(data_.size(1))][:16]#_iii的取值范围[,127]
                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]),win=u'origin_poem')
                #在visdom中输出这些句子（诗歌）中的前１６个
                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry =  ''.join(generate(model,word,ix2word,word2ix))
                    gen_poetries.append(gen_poetry)
                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]),win=u'gen_poem')  
        
        t.save(model.state_dict(),'%s_%s.pth' %(opt.model_prefix,epoch))

def gen(**kwargs):
    '''
    提供命令行接口，用以生成相应的诗
    '''

    for k,v in kwargs.items():
        setattr(opt,k,v)
    data,word2ix,ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256)#(vocab_size, embedding_dim, hidden_dim):
    map_location = lambda s,l:s
    state_dict = t.load(opt.model_path,map_location=map_location)
    model.load_state_dict(state_dict)
    
    if opt.use_gpu:
        model.cuda()
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
             start_words = opt.start_words
             prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode('utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None
     
     
    start_words= start_words.replace(',',u'，')\
                                     .replace('.',u'。')\
                                     .replace('?',u'？')

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model,start_words,ix2word,word2ix,prefix_words)
    print(''.join(result))

if __name__ == '__main__':
    import fire
    fire.Fire()
