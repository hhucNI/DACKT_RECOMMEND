import os
import pandas as pd


def process_submissions_folder(submissions_folder):
    result_rows = []
    for filename in os.listdir(submissions_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(submissions_folder, filename)
            df = pd.read_csv(file_path)
            df = df.sort_values('timestamp')

            submission_count = len(df)
            result_rows.append([submission_count])

            problem_ids = df['problem_id'].tolist()
            result_rows.append(problem_ids)

            submission_ids = df['submission_id'].tolist()
            result_rows.append(submission_ids)

            results = df['result'].tolist()
            result_rows.append(results)
    return result_rows


def save_to_csv(result_rows, output_file):
    # 创建DataFrame并保存为CSV文件
    df = pd.DataFrame(result_rows)
    df.to_csv(output_file, index=False, header=False)


if __name__ == "__main__":
    submissions_folder = 'submissions'
    output_file = 'combined_user_submissions.csv'

    # 处理提交文件夹并获取结果行
    result_rows = process_submissions_folder(submissions_folder)

    # 保存数据到CSV文件
    save_to_csv(result_rows, output_file)
    print(f"数据已保存到 {output_file}")