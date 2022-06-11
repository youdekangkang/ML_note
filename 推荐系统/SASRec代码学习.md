# Self-Attentive Sequential Recommendation

- [Wang-Cheng Kang](https://www.semanticscholar.org/author/Wang-Cheng-Kang/2741053), [Julian McAuley](https://www.semanticscholar.org/author/Julian-McAuley/35660011)
- Published 20 August 2018
- Computer Science
- 2018 IEEE International Conference on Data Mining (ICDM)

[Toc]

## 文章

**摘要**

Sequential dynamics are a key feature of many modern recommender systems, which seek to capture the 'context' of users' activities on the basis of actions they have performed recently. To capture such patterns, two approaches have proliferated: Markov Chains (MCs) and Recurrent Neural Networks (RNNs). Markov Chains assume that a user's next action can be predicted on the basis of just their last (or last few) actions, while RNNs in principle allow for longer-term semantics to be uncovered. Generally speaking, MC-based methods perform best in extremely sparse datasets, where model parsimony is critical, while RNNs perform better in denser datasets where higher model complexity is affordable. The goal of our work is to balance these two goals, by proposing a self-attention based sequential model (SASRec) that allows us to capture long-term semantics (like an RNN), but, using an attention mechanism, makes its predictions based on relatively few actions (like an MC). ==At each time step, SASRec seeks to identify which items are 'relevant' from a user's action history, and use them to predict the next item.== Extensive empirical studies show that our method outperforms various state-of-the-art sequential models (including MC/CNN/RNN-based approaches) on both sparse and dense datasets. Moreover, the model is an order of magnitude more efficient than comparable CNN/RNN-based models. Visualizations on attention weights also show how our model adaptively handles datasets with various density, and uncovers meaningful patterns in activity sequences.

文章出处：[1808.09781.pdf (arxiv.org)](https://arxiv.org/pdf/1808.09781.pdf)

代码：[kang205/SASRec: SASRec: Self-Attentive Sequential Recommendation (github.com)](https://github.com/kang205/SASRec)



实验环境(tf1.x)：

python >= 3.6

tensorflow >= 1.14.0

cuda >= 10.1

cudnn == 7.6.0

numpy == 1.16.0 



## 数据预处理

输入的数据格式是二元组和(userID,itemID)这样的格式输入的，经过处理之后能够变成序列模式：

```python
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)   # 最后输出的用户也是list类型的
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    # 构建测试集，训练集以及验证集
    # 最后一个交互的物品作为测试集，倒数第二个交互的物品作为验证集，其他作为训练集
    for user in User:
        nfeedback = len(User[user])
        # 如果用户序列太短，则不考虑把这个用户作为测试集
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]
```

这里是处理之后的数据：

![image-20220608164134716](SASRec代码学习.assets/image-20220608164134716.png)



## 模型的实现

#### 数据输入

作者通过采样采样一个batch_size的序列，使用了一个自定义的WarpSampler的类：

```python
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
```

采用并行处理的方式进行输入：

```python
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
```

采样操作：

```python
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        # 从用户集中随机采样一个用户
        # 若在训练集中该用户的序列长度小于1 则重新进行采样
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        # seq为序列,pos为用于存储交互的正样本,neg为用于1存储交互的负样本
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        # nxt存储最后一个交互物品
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        # reversed是逆序搜索，这里的i指的是交互的物品
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            # 保证当序列长度没到达maxlen时，正样本序列会补充为0，那么构成的负样本序列也应该是0
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))
```



#### 内核

基本的embedding层都是直接使用内置函数就能够完成的,包括item embedding和position embedding都是十分方便的

```python
# TODO: loss += args.l2_emb for regularizing embedding vectors during training
# https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
self.attention_layers = torch.nn.ModuleList()
self.forward_layernorms = torch.nn.ModuleList()
self.forward_layers = torch.nn.ModuleList()	
```

函数**log2feat**这个函数用于训练模型的部分内容：

```python
    def log2feats(self, log_seqs):
        # 取出物品对应的embedding值
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # 乘以一个embedding维度的根号值，感觉有点类似于attention的计算除以根号d那样
        seqs *= self.item_emb.embedding_dim ** 0.5
        # 构建位置矩阵
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        # 取出位置对应的embedding
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # 如果序列太短，前面等于0，那么对应的序列embedding也要等于0
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        # 实现论文中提到的前面物品在预测时不能用到后面物品的信息，需要使用mask来实现
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # 送入模型 进行预测
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats
```



**训练部分：**

```python
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
```



**评估部分：**

```python
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
    
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
```

生成模型文件：

```python
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
```



