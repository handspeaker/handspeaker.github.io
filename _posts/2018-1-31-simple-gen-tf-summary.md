---
layout: post
title: 记录tensorflow summary的简单方法
---

虽然tf官方希望用户把train、val程序分开写，但实际开发中，明显写在一起比较简单舒服，下面讨论如何在这种情况下记录train/val的summary

假设训练时的主要代码结构如下：

	loss_summary = ...
	other_summary = ...
	train_summaries = tf.summary.merge([loss_summary, other_summary])
	for i in range(self.batch_num):
		batching data...
		...
		... step ... train_summary_results = sess.run(... , train_summaries)
		train_file_writer.add_summary(train_summary_results)
		if step % self.save_inter == 0:
			... # save checkpoint
		if step % self.disp_inter == 0:
			... # display training process
    	if step % self.test_inter == 0:
			... # run model on test data
	

在保存训练的summary时很简单，每次训练的数据都是一个batch，只需要按照上面的写法，执行`train_file_writer.add_summary(train_summary_results)`即可。

但在test的时候，一般都是在整个测试集上获取summary，即需要对所有batch上获取的summary做平均。假设testset数量是1000，每个batch是50，那么test需要跑20次，每次都获取loss、准确率等指标，然后计算loss、准确率的平均数，才是整个testset的实际指标，最后按照summary格式写入到文件中。

但是在设计模型结构的时候，无论你是train还是test，结构都是一样的，每次只能计算一个batch的loss、准确率，没法单独为testset做平均。于是我想到了如下投机取巧的方法：

	test_summaries = tf.Summary()
	loss_val = test_summaries.value.add()
	loss_val.tag = 'loss'
	loss_val.simple_value = average_loss / batch_num
	acc_val = test_summaries.value.add()
	acc_val.tag = 'accuracy'
	acc_val.simple_value = average_accuracy / batch_num
	test_file_writer.add_summary(test_summaries, step)


其实就是自己创建一个`test_summariess`，把需要的东西填进去，模仿利用sess.run生成的`train_summary_results`。大家如果感兴趣可以把`train_summary_results`打印出来，其实就是这么个结构。目前我只保存过`scalar`，但是其他值应该也可以这么保存。
