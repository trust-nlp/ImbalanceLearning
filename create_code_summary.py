# 保存为 create_code_summary.py
import os
import sys
import argparse # 导入 argparse 用于处理命令行参数

# --- 核心功能函数 ---

def find_code_files(root_dir, code_extensions):
    """
    递归查找指定目录下所有符合扩展名的代码文件。

    Args:
        root_dir (str): 要搜索的根目录路径。
        code_extensions (tuple): 包含代码文件扩展名的元组 (e.g., ('.py', '.sh')).

    Returns:
        list: 包含所有找到的代码文件完整路径的列表。
    """
    code_files = []
    print(f"开始在 '{root_dir}' 及其子目录中搜索扩展名为 {code_extensions} 的文件...")
    if not os.path.isdir(root_dir):
        print(f"错误：指定的路径 '{root_dir}' 不是一个有效的目录。")
        return []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件扩展名是否在指定的代码扩展名列表中
            if filename.lower().endswith(code_extensions):
                full_path = os.path.join(dirpath, filename)
                code_files.append(full_path)
                # print(f"  找到: {full_path}") # 可选：打印找到的每个文件

    print(f"搜索完成，共找到 {len(code_files)} 个符合条件的文件。")
    return code_files

def write_code_to_txt(code_files, output_filename="code2txt.txt"):
    """
    将代码文件的路径和内容写入指定的文本文件。

    Args:
        code_files (list): 包含代码文件完整路径的列表。
        output_filename (str): 输出的 txt 文件名。
    """
    print(f"开始将代码写入到 '{output_filename}'...")
    written_count = 0
    error_count = 0
    try:
        # 使用 'w' 模式（写入），如果文件已存在则覆盖
        # 使用 utf-8 编码以支持多种字符
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for file_path in code_files:
                try:
                    # 写入文件路径作为分隔符和标识
                    outfile.write(f"--- 文件路径: {file_path} ---\n\n")

                    # 读取代码文件内容
                    # 同样使用 utf-8 尝试读取，如果失败则忽略错误字符
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        content = infile.read()
                        outfile.write(content)

                    # 在文件内容后添加换行符，以便区分不同文件
                    outfile.write("\n\n")
                    written_count += 1
                    # print(f"  已写入: {file_path}") # 可选：打印写入的文件

                except Exception as e:
                    error_msg = f"  错误：无法读取或写入文件 '{file_path}' - {e}\n"
                    print(error_msg)
                    outfile.write(f"--- 文件路径: {file_path} ---\n")
                    outfile.write(f"*** 错误：无法读取此文件内容 - {e} ***\n\n")
                    error_count += 1

        print(f"写入完成。成功写入 {written_count} 个文件，读取/写入失败 {error_count} 个文件。")
        print(f"结果已保存到：{os.path.abspath(output_filename)}")
        return os.path.abspath(output_filename) # 返回输出文件的绝对路径

    except IOError as e:
        print(f"严重错误：无法打开或写入输出文件 '{output_filename}' - {e}")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None

# --- 新增：封装主要逻辑的函数 ---
def generate_code_summary(target_dir, output_filename="code2txt.txt", allowed_extensions=('.py', '.sh')):
    """
    查找指定目录下的特定代码文件，并将它们的路径和内容写入文本文件。

    Args:
        target_dir (str): 要扫描的根目录路径。
        output_filename (str, optional): 输出的 txt 文件名。 Defaults to "code2txt.txt".
        allowed_extensions (tuple, optional): 允许的文件扩展名元组。 Defaults to ('.py', '.sh').

    Returns:
        str or None: 成功则返回输出文件的绝对路径，失败则返回 None。
    """
    # 查找所有符合条件的代码文件
    found_files = find_code_files(target_dir, allowed_extensions)

    # 如果找到了文件，则写入 txt
    if found_files:
        output_path = write_code_to_txt(found_files, output_filename)
        return output_path
    else:
        print(f"在 '{target_dir}' 目录下没有找到扩展名为 {allowed_extensions} 的文件。")
        return None

# --- 主程序入口 (用于命令行调用) ---
if __name__ == "__main__":
    # --- 配置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="查找指定目录下的 .py 和 .sh 文件，并将它们的路径和内容写入一个文本文件。")
    parser.add_argument("target_directory", help="要扫描代码的根目录路径。")
    parser.add_argument("-o", "--output", default="code2txt.txt",
                        help="输出的 txt 文件名 (默认为 'code2txt.txt')。")
    # 注意：这里我们直接在 generate_code_summary 函数中使用了默认的 .py 和 .sh 扩展名
    # 如果需要从命令行指定扩展名，可以取消下面这行的注释并修改 generate_code_summary 的调用
    # parser.add_argument("-e", "--extensions", nargs='+', default=['.py', '.sh'], help="要包含的文件扩展名列表 (默认为 .py .sh)。")

    # 解析命令行参数
    args = parser.parse_args()

    # --- 执行主要逻辑 ---
    # 调用封装好的函数，传入从命令行获取的参数
    # extensions_tuple = tuple(args.extensions) # 如果启用了命令行扩展名参数，则使用这行
    generate_code_summary(args.target_directory, args.output) #, extensions_tuple) # 如果启用了命令行扩展名参数，则传入