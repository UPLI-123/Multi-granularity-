from buildGraularUI import Ui_buildGraular
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from utils import is_comma_separated_numbers, build_Graular, build_Graular_Value
import numpy as np



class BuildGrau(QWidget, Ui_buildGraular):

    # 自定义传递的消息的类型，默认传递过去的是list集合
    # 第一个是视图信息，另一个是层次信息
    subWindoSignel = pyqtSignal(str, list, list, list, bool, list)

    def __init__(self):
        super(BuildGrau, self).__init__()
        self.setupUi(self)
        self.data_path = ''
        pass

    #  关闭当前窗口
    def cancelWindow(self):
        self.close()
        pass

    # 检测一下配置信息是否符合要求
    def okConfig(self):
        # 首先对当前窗口中的视图划分、层次划分来进行验证
        view_div = self.textViewDiv.toPlainText()
        # print(view_div)
        ci_div = self.textLevelDiv.toPlainText()
        # print(ci_div)
        # print(is_comma_separated_numbers(str(view_div)))
        if not is_comma_separated_numbers(view_div):
            QMessageBox.warning(self, '警告', '视图输入格式不正确！', QMessageBox.Close)
            return
            pass
        #  将视图信息转化为list集合，并进行计算判断一下是否与判断属性个数相等
        self.abs_len = int(self.textCount.toPlainText())
        view_div = [int(s) for s in view_div.split(',')]
        # print(view_div)
        # print(sum(view_div))
        if self.abs_len != sum(view_div):
            QMessageBox.warning(self, '警告', '视图没有完全划分！', QMessageBox.Close)
            return
            pass
        # 首先判断一下是否层次的划分是否和视图一致
        ci_div = ci_div.split("\n")
        # print(ci_div)
        if len(ci_div) != len(view_div):
            QMessageBox.warning(self, '警告', '没有对每个视图进行层次划分！', QMessageBox.Close)
            return
            pass
        #  对每一个视图的层次划分进行判断
        for i, st in enumerate(ci_div):
            if not is_comma_separated_numbers(st):
                QMessageBox.warning(self, '警告', '层次输入格式不正确！', QMessageBox.Close)
                return
                pass
            ci_st = [int(s) for s in st.split(",")]
            #  判断一下数值是否与视图的划分值一致
            if sum(ci_st) != view_div[i]:
                QMessageBox.warning(self, '警告', '视图的层次划分不正确！', QMessageBox.Close)
                return
                pass
            pass

        #  验证条件都满足后 调用粒结构构造方法 构造多视角、多层次粒结构
        data_info, view_index, ci_b_list, res_list, ci_list = build_Graular(view_div, ci_div, self.df2)
        # 将返回的结果构建成为字符串返回给父窗口
        # 传递信息
        self.subWindoSignel.emit(data_info, view_index, ci_list, res_list, False, ci_b_list)
        self.close()
        pass

    #  todo 导入层次划分的配置文件
    def impCi(self):
        #  选择文件弹窗
        filename, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*)")
        # print(filename)
        if filename == '' or filename is None:
            return
            pass
        # print(filename.split(".")[-1])
        if filename.split(".")[-1] != "txt":
            QMessageBox.information(self, '提示信息', '请添加正确的配置文件.txt！', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
            pass
        self.data_path = filename
        self.textCiRoot.setEnabled(True)
        filename = filename.split('/')[-1]
        # print(filename)
        self.textCiRoot.setText(filename)
        self.textCiRoot.setEnabled(False)
        QMessageBox.information(self, '提示信息', '添加成功！', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        self.btnImpConfig.setEnabled(False)
        self.btnResetCi.setEnabled(True)
        pass

    #  todo 清空导入文件的信息
    def resetCi(self):
        # 清空地址
        self.data_path = ''
        self.textCiRoot.setEnabled(True)
        self.textCiRoot.setText("")
        self.textCiRoot.setEnabled(False)
        self.btnImpConfig.setEnabled(True)
        self.btnResetCi.setEnabled(False)
        pass

    # todo 提交配置的文件
    def okCi(self):
        # print(self.data_path)
        if self.data_path == '' or self.data_path is None:
            QMessageBox.warning(self, '警告', '请先导入配置文件！', QMessageBox.Close)
            return
            pass
        #  1.读取配置文件
        with open(self.data_path, 'r') as f:
            data = f.read()
            pass
        # print(data)
        data = data.split("\n")
        # print(data)
        # 首先对当前窗口中的视图划分、层次划分来进行验证
        view_div = self.textViewDiv.toPlainText()
        if not is_comma_separated_numbers(view_div):
            QMessageBox.warning(self, '警告', '视图输入格式不正确！', QMessageBox.Close)
            return
            pass
        #  将视图信息转化为list集合，并进行计算判断一下是否与判断属性个数相等
        self.abs_len = int(self.textCount.toPlainText())
        view_div = [int(s) for s in view_div.split(',')]
        # print(view_div)
        # print(sum(view_div))
        if self.abs_len != sum(view_div):
            QMessageBox.warning(self, '警告', '视图没有完全划分！', QMessageBox.Close)
            return
            pass
        nums = data.count('----')
        if nums+1 != len(view_div):
            QMessageBox.warning(self, '警告', '配置信息和划分不一致！', QMessageBox.Close)
            return
            pass
        list_view = []
        view = []
        for d in data:
            if d == '----':
                list_view.append(view.copy())
                view = []
                continue
                pass
            ci_list = d.strip().split("|")
            view_abs = []
            for ci in ci_list:
                c = ci.split(',')
                abs_list = []
                for t in c:
                    t = t.split('-')
                    abs_list.append(float(t[0]))
                    abs_list.append(float(t[1]))
                    pass
                view_abs.append(abs_list.copy())
                pass
            view.append(view_abs.copy())
            pass
        list_view.append(view.copy())
        # print(list_view)
        # data_info, view_index,ci_list, res_list, view_atr = build_Graular_Value(list_view, self.df)
        data_info, view_index, ci_list, res_list, view_atr = build_Graular_Value(list_view, self.df, self.df2)
        self.subWindoSignel.emit(data_info, view_index, ci_list, res_list, True, view_atr)
        self.close()
        pass

    #  todo 初始化数据
    def initData(self, df):
        self.df = df
        # print(df)
        fn = len(df.columns) - 1  # 染色体长度
        obn = len(df)  # 数据集样本个数
        # print(fn, obn)
        df1 = df.iloc[0:obn, 0:fn]
        self.df1 = df1
        # print(df1)
        df2 = np.mat(df1)
        # print(df2)
        self.df2 = df2
        pass



