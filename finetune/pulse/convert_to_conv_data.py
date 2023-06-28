import argparse
import json
import datetime
import os

'''
orig_data: [
    {
        "instruction": "##结构化任务##根据下文中信息，判断主动脉根部内径是什么？请提取文中对应的值",
        "input": "检查途径：经体表  图像等级：乙  检查项目：二维     M型     彩色    多普勒（脉冲式    连续式）一、M型主要测值（单位mm)：                            名称          测量值    正常参考值    主动脉根部内径      39        20-37    左房内径            36        19-40    左室舒张末期内径    47        35-56    左室收缩末期内径    28        20-37    室间隔厚度          11        6-11    左室后壁厚度        10        6-11   二、二维超声心动图描述：1.各房室无明显扩大，主动脉根部稍增宽。2.各心瓣膜未见明显增厚，开放不受限。3.左室壁不增厚，静息状态下左室各节段收缩活动未见明显异常。三、彩色多普勒超声描述:1.各心瓣膜未见明显异常反流。2.舒张期经二尖瓣口血流频谱：E/A＜1。舒张期经二尖瓣口血流：E=54cm/s,A=84cm/s。3.房、室间隔水平未见明显异常分流信号。四、左心功能测定:         名称                    测量值    左室舒张末期容量（ml）        102     左室收缩末期容量（ml）        29     每搏输出量（ml）              74     左室短轴缩短率（%）           41     左室射血分数（%）             72 五、组织多普勒检查：    二尖瓣瓣环水平：室间隔侧： e'=6cm/s， E/e'=9。                    左室侧壁： e'=9cm/s,  E/e'=6。",
        "output": "39"
    },
    {
        "instruction": "##结构化任务##根据下文中信息，判断主动脉根部内径是什么？请提取文中对应的值",
        "input": "检查途径：经体表  图像等级：乙  检查项目：二维     M型     彩色    多普勒（脉冲式    连续式）一、M型主要测值（单位mm)：                            名称          测量值    正常参考值    主动脉根部内径      38        20-37    左房内径            44        19-40    左室舒张末期内径    52        35-56    左室收缩末期内径    33        20-37    室间隔厚度          13        6-11    左室后壁厚度        13        6-11   二、二维超声心动图描述：1.左房增大,主动脉根部内径增宽。2.各心瓣膜未见明显增厚，开放不受限。3.左室壁增厚，静息状态下左室各节段收缩活动未见明显异常。三、彩色多普勒超声描述:1.各心瓣膜示轻度微主动脉瓣反流。2.舒张期经二尖瓣口血流频谱：E/A＜1。舒张期经二尖瓣口血流：E=63cm/s,A=91cm/s。3.房、室间隔水平未见明显异常分流信号。四、左心功能测定:         名称                    测量值    左室舒张末期容量（ml）        129     左室收缩末期容量（ml）        45     每搏输出量（ml）              84     左室短轴缩短率（%）           36     左室射血分数（%）             65 五、组织多普勒检查：    二尖瓣瓣环水平：室间隔侧： e'=5cm/s， E/e'=12.6。                    左室侧壁： e'=13cm/s,  E/e'=4.9。",
        "output": "38"
    }]
convert: {
    "id": xxx,
    "conversations":[
        {"from": "human", "value": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？"},
        {"from": "assistant", "value": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"},
    ]
}
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_data",
    )
    parser.add_argument(
        "--write_data",
    )
    parser.add_argument(
        "--dataset_name",
    )
    args = parser.parse_args()
    f_write = open(args.write_data,"w")
    with open(args.orig_data, 'r', encoding="utf-8") as f:
        datas = json.load(f)
        num_id = 1
        for data in datas:
            conversations = [{"from": "human", "value": data['instruction']+'\n'+data['input']},{"from": "assistant", "value": data['output']}]
            uniq_id = data['id'] if "id" in data else args.dataset_name+"-"+str(num_id)
            item = {"id":uniq_id, "conversations": conversations}
            f_write.write(json.dumps(item, ensure_ascii=False)+"\n")
            num_id += 1
    f_write.close()


if __name__ == "__main__":
    main()