---
layout: post
title: 记录tensorflow summary的简单方法
---

虽然tf官方希望用户把 _train_ , _val_ 程序分开写，但实际开发中，明显写在一起比较简单舒服，但在保存数据到 _summary_ 时， _val_ 部分和 _train_ 部分不太一样，会有一些问题，下面讨论如何在这种情况下记录 _train/val_ 的 _summary_ 。

假设训练时的主要代码结构如下：

```python
loss_summary = ...
other_summary = ...
train_summaries = tf.summary.merge([loss_summary, other_summary])
for i in range(self.batch_num):
	batching data...
	...
	...
	... step ... train_summary_results = sess.run(... , train_summaries)
	train_file_writer.add_summary(train_summary_results)
	if step % self.save_inter == 0:
		... # save checkpoint
	if step % self.disp_inter == 0:
		... # display training process
    if step % self.test_inter == 0:
		... # run model on val data
```	

保存 _train_ 部分的 _summary_ 很简单，tf的示例代码也给了很多，先利用 _sess.run_ 计算出 _train\_summary\_results_ ，即当前 _batch_ 的统计数据，然后保存到文件

但在 _val_ 部分时，一般都在所有验证数据上获取 _loss_ , _accuracy_ 等 _summary_ 数据，再保存到文件。这样只有两种方法：

1. _val_ 部分的 _batch\_size_ 改为验证集大小  
2. _batch\_size_ 不变，对所有 _batch_ 上获取的 _loss_ , _accuracy_ 计算平均

第一种方法存在的问题是，如果验证集数据较大， _batch\_size_ 会设置的较大，可能会引起内存or显存溢出，这个没法解决。

第二种方法存在的问题是，没法按照train部分的做法做，因为要的是整个验证数据的平均值，而不是每个 _batch_ 的值,这个有办法解决。

在设计模型结构的时候，无论是 _train_ 还是 _val_ ，网络结构都是一样的，每次只能计算一个 _batch_ 的 _loss_ , _accuracy_ ，没法单独为验证集修改。于是我想到了如下投机取巧的方法：先利用循环计算验证集每个 _batch_ 的 _loss_ , _accuracy_ ，进行累加，记为 _average\_loss_ 和 _average\_accuracy_ ，然后进行如下操作：

```python
test_summaries = tf.Summary()
loss_val = test_summaries.value.add()
loss_val.tag = 'loss'
loss_val.simple_value = average_loss / batch_num
acc_val = test_summaries.value.add()
acc_val.tag = 'accuracy'
acc_val.simple_value = average_accuracy / batch_num
test_file_writer.add_summary(test_summaries, step)
```

其实就是自己创建一个 _test\_summaries_ ，把需要的东西填进去，模仿利用 _sess.run_ 生成的 _train\_summary\_results_ ，再保存到文件。大家如果感兴趣可以把 _train\_summary\_results_ 打印出来，其实就是这么个结构。目前我只保存过 _scalar_ ，但是其他值应该也可以这么保存。










