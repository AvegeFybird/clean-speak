#为文件提供绝对路径

import os

def get_project_root() -> str:
    """
    获取工程所在的根目录
    :return:
    """
    #当前文件的绝对路径
    current_path = os.path.abspath(__file__)
    #获取工程的根目录
    current_dir = os.path.dirname(current_path)


    project_root = os.path.dirname(current_dir)
    return project_root

def get_abs_path(relative_path: str):
    """
    获取绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root,relative_path)


if __name__ == '__main__':
    pass




