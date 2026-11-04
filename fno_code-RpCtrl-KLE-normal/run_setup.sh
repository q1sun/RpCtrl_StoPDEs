#!/bin/bash
#SBATCH -J RpCtrl-setup-s1e2     # 任务名字是 test
#SBATCH -o RpCtrl-setup-s1e2     # 输出, 目录./out必须存在，否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=tmp_sq          	 # qos(quality of service): normal or short or debug, 对应不同优先级及最大可用时长.
#SBATCH -N 1                  	 # 申请 1 个节点
#SBATCH --cpus-per-task=16    	 # 申请 4 个 cpu 核心
#SBATCH --mem=10G             	 # 申请10G内存
#SBATCH -p k80                	 # 申请 1 个节点
#SBATCH -t 1-23:59:00         	 # 申请Job运行时长0小时5分钟0秒,若要申请超过一天时间,如申请1天,书写格式为#SBATCH -t 1-00:00:00

module add matlab             	 # 添加 MATLAB 模块
# 使用 MATLAB 运行当前文件夹中的 Test.m 文件
matlab -nodesktop -nosplash -nodisplay -r "setup_KLE_normal"