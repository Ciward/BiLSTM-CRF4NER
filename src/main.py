from train.train import Train
from tools.get_ner_level_acc import precision

if __name__ == "__main__":

    use_pretrained_w2v = True
    model_type = "bilstm-crf"
    do_train = False  # 控制是否训练
    do_test = True    # 控制是否只加载模型并测试

    model_train = Train()

    if do_train:
        model_train.train(use_pretrained_w2v=use_pretrained_w2v, model_type=model_type)
    if do_test:
        model_train.load_and_test(use_pretrained_w2v=use_pretrained_w2v, model_type=model_type)

    text = "常建良，男，1963年出生，工科科学士，高级工程师，北京物资学院客座副教授。"

    result = model_train.predict(text, use_pretrained_w2v, model_type)
    print(result[0])

    result_dic = model_train.get_ner_list_dic(text, use_pretrained_w2v, model_type)
    print(result_dic)

