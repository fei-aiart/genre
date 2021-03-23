import os


class Data_Dictionary(object):

    def __init__(self, data_dic_path):
        self.dataset_dic = {}
        self.dataset_shape = {}
        self.dataset_labals = {}
        with open(data_dic_path, 'r') as file:
            for line in file:
                line = line.strip('\n')
                line_s = line.split("||")
                assert len(line_s) >= 5
                name = line_s[0]
                elem_ref_list = line_s[1:3]
                self.dataset_dic[name] = elem_ref_list
                h = int(line_s[3])
                w = int(line_s[4])
                self.dataset_shape[name] = (h, w)
                if len(line_s) == 6:  # with total style labels
                    self.dataset_labals[name] = int(line_s[5])

    def get_refpath_by_name(self, dataset_name, type):
        ref_list = self.dataset_dic[dataset_name]
        if type not in ["train", "test"]:
            raise KeyError("Your type name is wrong ,please choose in 'train' and 'test'")
        if not isinstance(ref_list, list):
            raise RuntimeError("Your file not contain both two of ref path")

        total = ""
        if type == "train":
            total = os.path.join(ref_list[0])

        elif type == "test":
            total = os.path.join(ref_list[1])

        return total

    def get_dataset_total_label(self, dataset_name):
        return self.dataset_labals[dataset_name]

    def get_dataset_img_size(self, dataset_name):
        return self.dataset_shape[dataset_name]
