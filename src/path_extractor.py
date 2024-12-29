import pandas as pd
import json
from anytree import Node, RenderTree
from anytree.search import findall_by_attr
from anytree.walker import Walker
import numpy as np
import argparse
import os
import random
import javalang
from config import *
import visualize
from tqdm import tqdm, trange

"""

处理code表，和preprocess独立
"""


# 扩展本函数，获取更多信息
def get_token(node):
    token = ''
    pos = "(null)"
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'  # node.pop()
    elif isinstance(node, javalang.ast.Node):
        token = node.__class__.__name__
        pos = node.position
    return token + "%" + str(pos)


def get_children(root):
    if isinstance(root, javalang.ast.Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    # flatten nested item
    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    ret = list(expand(children))
    return ret


# me
def get_position(cur_node):
    pass


def get_trees(current_node, parent_node, order):
    # token 是cur node，children是cur node的children
    # 本层处理current_node,下层处理child
    token, children = get_token(current_node), get_children(current_node)
    # name :  [order, token]
    node = Node([order, token], parent=parent_node, order=order)

    # 根据cur node 的children递归建树
    for child_order in range(len(children)):
        get_trees(children[child_order], node, order + str(int(child_order) + 1))


def get_path_length(path):
    """Calculating path length.
    Input:
    path: list. Containing full walk path.

    Return:
    int. Length of the path.
    """

    return len(path)


def get_path_width(raw_path):
    """Calculating path width.
    Input:
    raw_path: tuple. Containing upstream, parent, downstream of the path.

    Return:
    int. Width of the path.
    """
    # 最大折角

    # 头节点之下一层的折角，所以2也合理
    return abs(int(raw_path[0][-1].order) - int(raw_path[2][0].order))


def hashing_path(path, hash_table):
    """Calculating path width.
    Input:
    raw_path: tuple. Containing upstream, parent, downstream of the path.

    不存在的话创建

    Return:
    str. Hash of the path.
    """

    if path not in hash_table:
        hash = random.getrandbits(128)
        hash_table[path] = str(hash)
        return str(hash)
    else:
        return hash_table[path]


def get_node_rank(node_name, max_depth):
    """Calculating node rank for leaf nodes. 补0补满max_depth
    Input:
    node_name: list. where the first element is the string order of the node, second element is actual name.
    max_depth: int. the max depth of the code.


    Return:
    list. updated node name list.
    """
    while len(node_name[0]) < max_depth:
        node_name[0] += "0"
    return [int(node_name[0]), node_name[1]]


def extracting_path(java_code, max_length, max_width, hash_path, hashing_table, uc_count):
    """Extracting paths for a given json code.
    Input:
    json_code: json object. The json object of a snap program to be extracted.
    max_length: int. Max length of the path to be restained.
    max_width: int. Max width of the path to be restained.
    hash_path: boolean. if true, MD5 hashed path will be returned to save space.
    hashing_table: Dict. Hashing table for path.

    Return:
    walk_paths: list of AST paths from the json code.
    """

    # max_width 2
    # max_length 8
    # hash_path  True
    # hashing_table 初始为空

    # Initialize head node of the code.
    # java_code是一个大节点，get_token取得这个节点的token
    head = Node(["1", get_token(java_code)])

    # Recursively construct AST tree.
    # 对于所有children递归建立树，get_children把多重列表展平
    for child_order in range(len(get_children(java_code))):
        get_trees(get_children(java_code)[child_order], head, "1" + str(int(child_order) + 1))

    # Getting leaf nodes.
    # 第三方树操作工具
    leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)

    # visualize.draw(head)
    # visualize.exportDot(head)
    # raise
    # Getting max depth.
    max_depth = max([len(node.name[0]) for node in leaf_nodes])

    # Node rank modification.![](../../../Java-to-AST-with-Visualization/output/HelloWorld_caughtSpeeding.png)
    for leaf in leaf_nodes:
        # 补0 node.name[1] 补到max_depth(不是length)
        leaf.name = get_node_rank(leaf.name, max_depth)

    walker = Walker()
    text_paths = []
    # Walk from leaf to target
    # 遍历组合 i->(i+1-n)
    for leaf_index in range(len(leaf_nodes) - 1):
        for target_index in range(leaf_index + 1, len(leaf_nodes)):
            # def walk(self, start, end):

            start = leaf_nodes[leaf_index]
            end = leaf_nodes[target_index]
            raw_path = walker.walk(start, end)

            # Combining up and down streams
            # path 是一段+拐点+另一段 TODO 看看name是什么
            walk_path = [n.name[1] for n in list(raw_path[0])] + [raw_path[1].name[1]] + [n.name[1] for n in
                                                                                          list(raw_path[2])]
            text_path = "@".join(walk_path)

            # Only keeping satisfying paths.
            # get_path_length(walk_path) : len(path)
            if get_path_length(walk_path) <= max_length and get_path_width(raw_path) <= max_width:
                if not hash_path:
                    # If not hash path, then output original text path.
                    text_paths.append(
                        walk_path[0] + config.single_path_sep + text_path + config.single_path_sep + walk_path[-1])
                else:
                    # TODO 默认这边
                    # If hash, then output hashed path.
                    appended = walk_path[0] + config.single_path_sep + hashing_path(text_path,
                                                                                    hashing_table) + config.single_path_sep + \
                               walk_path[-1]
                    text_paths.append(appended)
    # if java_code=="Uncompilable":
    #     uc_count[0]=uc_count[0]+1
    #     print("Uncompilable")
    #     print(text_paths)
    return text_paths


def program_parser(func):
    # 重点，parse code
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    # tree=parser.parse_class_body_declaration()
    return tree


def main(use_all_assignments=False):
    """
     生成labeled
    """
    print(os.getcwd())
    main_df = pd.read_csv(config.post_preprocess_main_table)
    code_df = pd.read_csv(config.raw_code_table)
    compile_state = main_df[main_df["EventType"] == "Compile"][["CodeStateID", "Compile.Result"]]

    # 先不merge了吧，没必要
    # # 使用nunique函数统计不同的数据数量
    num_unique_values = compile_state['Compile.Result'].nunique()
    #
    # # 打印结果 检查是否只有这两种
    print(f'在列 "column_name" 中有 {num_unique_values} 种不同的数据。')

    # full_df=main_df
    main_df = main_df[main_df["EventType"] == "Run.Program"]
    assert compile_state.shape[0] == main_df.shape[0]
    main_df['compile_state'] = np.array(compile_state["Compile.Result"] == "Success").astype(int)
    if not use_all_assignments:
        main_df = main_df[main_df["AssignmentID"] == Config.ASSIGNMENT_ID]
    # main_df.to_csv("../data/only439.csv")
    # raise
    # main_df = main_df[main_df["CodeStateID"] == 1746834]
    # main_df = main_df[main_df["CodeStateID"] == 1746929]

    # me
    # main_df=main_df[main_df["EventType"]==]
    # EventType
    # 1746826

    # 修改表
    main_df["raw_score"]=main_df["Score"]
    main_df['Score'] = np.array(main_df["Score"] == 1).astype(int)

    # merge
    main_df = main_df.merge(code_df, left_on="CodeStateID", right_on="CodeStateID")

    parsed_code = []
    # is_pass_compilation = []
    # def check_if_pass_compilation(stu_id):
    #     full_df
    uncompilable_count = 0
    print(f"all code length in assignment {Config.ASSIGNMENT_ID}  : {main_df['Code'].shape}")
    # for c in tqdm(list(main_df['Code']), desc="generate AST"):
    for c, stu_di in tqdm(zip(list(main_df['Code']), list(main_df["SubjectID"])), desc="generate AST"):
        try:
            parsed = program_parser(c)
            # is_pass_compilation.append(1)
        except:
            parsed = "Uncompilable"
            uncompilable_count += 1
            # is_pass_compilation.append(0)

        parsed_code.append(parsed)

    print(f"Uncompilable Count  :   {uncompilable_count}   ===========")
    print(f"Uncompilable Count  :   {uncompilable_count}   ===========")
    print(f"Uncompilable Count  :   {uncompilable_count}   ===========")
    # Initialize hashing_table.
    hashing_table = {}
    # hashing_table = np.load("path_hashing_dict.npy",allow_pickle=True).item()

    # Extracting paths for all programs in the csv file. Output is [["start,path_hash/path,end",...,...],...].
    # AST_paths = [extracting_path(java_code, max_length=config.code_path_length, max_width=config.code_path_width, hash_path=True, hashing_table=hashing_table) for java_code in parsed_code]
    AST_paths = []
    uc_count = [0]
    for java_code in tqdm(parsed_code, desc="generate paths"):
        # max_width 2
        # max_length 8
        # hash_path  True
        # hashing_table 初始为空
        AST_paths.append(
            extracting_path(java_code, max_length=config.code_path_length, max_width=config.code_path_width,
                            hash_path=True, hashing_table=hashing_table, uc_count=uc_count))
    # Storing the raw paths

    # AST_paths 二维列表 ： 一个element（一个path的list）代表一段程序
    # 每个代码片段对应的所有path用@链接
    _path = [config.paths_sep.join(A) for A in AST_paths]
    main_df["RawASTPath"] = _path
    # 给main_df的一列赋值，每个元素对应一段代码t't

    # main_df.to_csv("../data/labeled_paths.csv", sep="\t", header=True)
    main_df.to_csv(os.path.join(config.processed_data_dir, "labeled_paths.tsv"), sep="\t", header=True)

    # 生成额外的带codebert vector的 per-assignment 的dataset
    # try:
    #     chunk_size = 128
    #     num_rows = len(df)
    #     tokenizer = AutoTokenizer.from_pretrained("../huggingface_cloned/codebert-base")
    #     # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base",cache_dir="../huggingface_pretrained/codebert_cache_tokenizer")
    #     model = AutoModel.from_pretrained("../huggingface_cloned/codebert-base")
    #     all_codebert_vectors=[]
    #     for i in range(0, num_rows, chunk_size):
    #         chunk = main_df.iloc[i:i + chunk_size]
    #         code_chunk=list(chunk["Code"])
    #         code_tokens = tokenizer.tokenize(code_chunk)
    #         code_tokens=[[tokenizer.cls_token]+code+[tokenizer.eos_token] for code in code_tokens]
    #         tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    #         temp = torch.tensor(tokens_ids)[None, :]
    #         context_embeddings = model(temp)[0]
    #         d={k:context_embeddings[i] for i,k in enumerate(chunk["CodeStateID"])}
    #         # all_codebert_vectors.extend(list(context_embeddings))
    #     np.save(os.path.join(config.processed_data_dir,"codeid_codevec.npy"),d)
    # except Exception:
    #     print("Fail to 生成额外的带codebert vector的 per-assignment 的dataset")


if __name__ == "__main__":
    main()
