import argparse
import os
from Evaluation.evaluator import Eval_thread  # 导入 Eval_thread 类
from Evaluation.dataloader import EvalDataset  # 确保导入 EvalDataset 类

#CAMO+COD10K+CHAMELEON+NC4K
def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Evaluation script for segmentation models")

    # 添加命令行参数，并设置默认值
    parser.add_argument('--save_test_path_root',
                        default='/home/zqq/桌面/twj/BBS-Net-master/dataTest/zoomnext_pvtv2b2/',  # 预测结果保存的根目录
                        type=str,
                        help='Path to the directory where saliency maps (predictions) are saved')

    parser.add_argument('--test_paths',
                        type=str,
                        default='CAMO+COD10K+CHAMELEON+NC4K',  # 您要评估的数据集
                        help='The test dataset paths, separated by "+"')

    # 是否进行评估
    parser.add_argument('--Evaluation',
                        default=True,  # 默认设置为 True
                        type=bool,
                        help='Set True for evaluation, False for no evaluation')

    parser.add_argument('--methods',
                        type=str,
                        default='',
                        help='The methods to evaluate, separated by "+"')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./',
                        help='Path for saving the result.txt')

    # 解析命令行参数
    args = parser.parse_args()

    # 如果设置了 Evaluation 为 True，执行评估
    if args.Evaluation:
        methods = args.methods.split('+')
        test_paths = args.test_paths.split('+')

        for test_path in test_paths:
            dataset_name = test_path

            for method in methods:
                # 预测结果路径：D:\result\数据集名\方法名\
                pred_dir_all = os.path.join(args.save_test_path_root, dataset_name, method)

                # GT数据路径：D:\DataSource\数据集名\GT\
                gt_dir_all = os.path.join('/home/zqq/桌面/twj/datasource/RGBDCOD/test/', dataset_name, 'mask')

                # 确保路径存在，如果路径不存在，可以打印提示或者跳过
                try:
                    loader = EvalDataset(pred_dir_all, gt_dir_all)  # 创建数据加载器
                    thread = Eval_thread(loader, method, test_path, args.save_dir, cuda=True)  # 创建评估线程
                    result = thread.run()  # 运行评估
                    print(result)  # 打印评估结果
                except FileNotFoundError as e:
                    print(f"Error: {e}. Check if the directory {pred_dir_all} exists.")
    else:
        print("Evaluation is turned off. No evaluation will be performed.")


if __name__ == "__main__":
    main()