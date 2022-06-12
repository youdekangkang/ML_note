# BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer

- [Fei Sun](https://www.semanticscholar.org/author/Fei-Sun/143770118), [Jun Liu](https://www.semanticscholar.org/author/Jun-Liu/46700431), [Jian Wu](https://www.semanticscholar.org/author/Jian-Wu/2115903429), [Changhua Pei](https://www.semanticscholar.org/author/Changhua-Pei/3438562), [Xiao Lin](https://www.semanticscholar.org/author/Xiao-Lin/2117690202), [Wenwu Ou](https://www.semanticscholar.org/author/Wenwu-Ou/10336865), [Peng Jiang](https://www.semanticscholar.org/author/Peng-Jiang/2061280682) less
- Published 14 April 2019
- Computer Science
- Proceedings of the 28th ACM International Conference on Information and Knowledge Management 

 ## 文章

**摘要**

Modeling users' dynamic preferences from their historical behaviors is challenging and crucial for recommendation systems. Previous methods employ sequential neural networks to encode users' historical interactions from left to right into hidden representations for making recommendations. Despite their effectiveness, we argue that such left-to-right unidirectional models are sub-optimal due to the limitations including: unidirectional architectures restrict the power of hidden representation in users' behavior sequences; \item they often assume a rigidly ordered sequence which is not always practical. \end enumerate* To address these limitations, we proposed a sequential recommendation model called BERT4Rec, which employs the deep bidirectional self-attention to model user behavior sequences. `To avoid the information leakage and efficiently train the bidirectional model, we adopt the Cloze objective to sequential recommendation, predicting the random masked items in the sequence by jointly conditioning on their left and right context.` In this way, we learn a bidirectional representation model to make recommendations by allowing each item in user historical behaviors to fuse information from both left and right sides. Extensive experiments on four benchmark datasets show that our model outperforms various state-of-the-art sequential models consistently.

文章出处：https://arxiv.org/pdf/1904.06690.pdf

代码出处：[FeiSun/BERT4Rec: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer (github.com)](https://github.com/FeiSun/BERT4Rec)



关于如何阅读代码，我们需要关注一下模型的输入，模型输出以及运行函数的本体

## 代码阅读
### 模型输入

首先从Bertconfig这个文件中就可以看到模型的基本参数：

![image-20220611232848329](D:\OneDrive\PAT_ACM\笔记区\机器学习\推荐系统\Bert4Rec代码学习.assets\image-20220611232848329.png)

这一堆代码在`run.py`中被这样调用,也就是说这个位置代表了模型的初始化位置：

```python
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
```

以下这一段函数用于输入数据的训练，这里有一个estimator类，比较复杂不用过分研究，只需要知道这东西给i是拿来评估模型效果就好了：

```python
# If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": FLAGS.batch_size
        })

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(
            input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_input_fn = input_fn_builder(
            input_files=test_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        #tf.logging.info('special eval ops:', special_eval_ops)
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=None,
            hooks=[EvalHooks()])
```



### 模型的输出

需要注意的是，因为在模型内部数据都是以张量的形式存在的，所以需要做一定的转化才能以Python的数据类型进行输出。

首先我们研究EvalHooks()这个函数，内部一共有四个小函数:begin,end,before_run,after_run

**begin**

比较简单，做一个各种数据的初始化准备

```python
    def begin(self):
        self.valid_user = 0.0

        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.ap = 0.0

        np.random.seed(12345)

        self.vocab = None

        if FLAGS.user_history_filename is not None:
            print('load user history from :' + FLAGS.user_history_filename)
            with open(FLAGS.user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if FLAGS.vocab_filename is not None:
            print('load vocab from :' + FLAGS.vocab_filename)
            with open(FLAGS.vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            # print(values)
            sum_value = np.sum([x for x in values])
            # print(sum_value)
            self.probability = [value / sum_value for value in values]
```

**end**

end里面有一个session输入，session.run就是tensorflow提供给用户进行张量转python类型的函数。

```python
    def end(self, session):
        print(
            "ndcg@1:{}, hit@1:{}， ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{}".
            format(self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
                   self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
                   self.ndcg_10 / self.valid_user,
                   self.hit_10 / self.valid_user, self.ap / self.valid_user,
                   self.valid_user))
```

以下两个函数提供张量和python数据类型之间的转换

**before_run**

before_run中放入run_context（你需要转换的内容）after_run中run_values输出python数据类型。

一般先把需要转换的输出内容放到集合里面去

```python
    def before_run(self, run_context):
        variables = tf.get_collection('eval_sp')
        return tf.train.SessionRunArgs(variables)
```



**after_run**

```python
    def after_run(self, run_context, run_values):
        #tf.logging.info('run after run')
        #print('run after run')
        masked_lm_log_probs, input_ids, masked_lm_ids, info = run_values.results
        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, FLAGS.max_predictions_per_seq, masked_lm_log_probs.shape[1]))
#         print("loss value:", masked_lm_log_probs.shape, input_ids.shape,
#               masked_lm_ids.shape, info.shape)

        for idx in range(len(input_ids)):
            rated = set(input_ids[idx])
            rated.add(0)
            rated.add(masked_lm_ids[idx][0])
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(info[idx][0])][0])
            item_idx = [masked_lm_ids[idx][0]]
            # here we need more consideration
            masked_lm_log_probs_elem = masked_lm_log_probs[idx, 0]  
            size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
            if FLAGS.use_pop_random:
                if self.vocab is not None:
                    while len(item_idx) < 101:
                        sampled_ids = np.random.choice(self.ids,101,replace=False,p=self.probability)
                        sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                        item_idx.extend(sampled_ids[:])
                    item_idx = item_idx[:101]
            else:
                # print("evaluation random -> ")
                for _ in range(100):
                    t = np.random.randint(1, size_of_prob)
                    while t in rated:
                        t = np.random.randint(1, size_of_prob)
                    item_idx.append(t)

            predictions = -masked_lm_log_probs_elem[item_idx]
            rank = predictions.argsort().argsort()[0]

            self.valid_user += 1

            if self.valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            if rank < 1:
                self.ndcg_1 += 1
                self.hit_1 += 1
            if rank < 5:
                self.ndcg_5 += 1 / np.log2(rank + 2)
                self.hit_5 += 1
            if rank < 10:
                self.ndcg_10 += 1 / np.log2(rank + 2)
                self.hit_10 += 1

            self.ap += 1.0 / (rank + 1)
```



### 模型体内部

embedding：

- word_embeddings
- position_embeddings
- token_type_embeddings

l embedding类型use_one_hot_embeddings、tf.nn.embedding_lookup区别

 

**transformer_model**

预定义的一些参数

```python
def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
```

从这里看到模型实现attenton的部分代码：

```python
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output
            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=
                        attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)
```

num_hidden_layers表示transformer经过了多少层，而tf.variable_scope则制定了变量的作用域

而进入attention以后它是给每一个KQV都做了一个全连接层。防止KQV完全一致过拟合，有些attention写法是对KQV分别进行一个dropout

```python
# `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))
```



mask的处理流程也非常重要:

用一个提前维护好的attention_mask矩阵0-1，如果是0给用一个负-10000的数字代替加在attention_scores后面 softmax是ex,所以x趋向于负无穷得分就是0。

```python
    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32))*(-10000.0)

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder
```



自定义去屏蔽特殊位置的代码:



