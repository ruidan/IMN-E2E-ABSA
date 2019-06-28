import numpy as np 

def convert_to_list(y_aspect, y_sentiment, mask):
    y_aspect_list = []
    y_sentiment_list = []
    for seq_aspect, seq_sentiment, seq_mask in zip(y_aspect, y_sentiment, mask):
        l_a = []
        l_s = []
        for label_dist_a, label_dist_s, m in zip(seq_aspect, seq_sentiment, seq_mask):
            if m == 0:
                break
            else:
                l_a.append(np.argmax(label_dist_a))
                ### all entries are zeros means that it is a background word or word with conflict sentiment
                ### which are not counted for training AS
                ### also when evaluating, we do not count conflict examples
                if not np.any(label_dist_s):
                    l_s.append(0)
                else:
                    l_s.append(np.argmax(label_dist_s)+1)
        y_aspect_list.append(l_a)
        y_sentiment_list.append(l_s)
    return y_aspect_list, y_sentiment_list


def score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, train_op):
    if train_op:
        begin = 3
        inside = 4
    else:
        begin = 1
        inside = 2

        # predicted sentiment distribution for aspect terms that are correctly extracted
        pred_count = {'pos':0, 'neg':0, 'neu':0}
        # gold sentiment distribution for aspect terms that are correctly extracted
        rel_count = {'pos':0, 'neg':0, 'neu':0}
        # sentiment distribution for terms that get both span and sentiment predicted correctly
        correct_count = {'pos':0, 'neg':0, 'neu':0}
        # sentiment distribution in original data
        total_count = {'pos':0, 'neg':0, 'neu':0}

        polarity_map = {1: 'pos', 2: 'neg', 3: 'neu'}

        # count of predicted conflict aspect term
        predicted_conf = 0

    correct, predicted, relevant = 0, 0, 0

    for i in range(len(true_aspect)):
        true_seq = true_aspect[i]
        predict = predict_aspect[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == begin:
                relevant += 1
                if not train_op:
                    if true_sentiment[i][num]!=0:
                        total_count[polarity_map[true_sentiment[i][num]]]+=1
                     
                if predict[num] == begin:
                    match = True 
                    for j in range(num+1, len(true_seq)):
                        if true_seq[j] == inside and predict[j] == inside:
                            continue
                        elif true_seq[j] != inside  and predict[j] != inside:
                            break
                        else:
                            match = False
                            break

                    if match:
                        correct += 1
                        if not train_op:
                            # do not count conflict examples
                            if true_sentiment[i][num]!=0:
                                rel_count[polarity_map[true_sentiment[i][num]]]+=1
                                pred_count[polarity_map[predict_sentiment[i][num]]]+=1
                                if true_sentiment[i][num] == predict_sentiment[i][num]:
                                    correct_count[polarity_map[true_sentiment[i][num]]]+=1

                            else:
                                predicted_conf += 1



        for pred in predict:
            if pred == begin:
                predicted += 1

    p_aspect = correct / (predicted + 1e-6)
    r_aspect = correct / (relevant + 1e-6)
    # F1 score for aspect (opinion) extraction
    f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

    acc_s, f_s, f_absa = 0, 0, 0

    if not train_op:
        num_correct_overall = correct_count['pos']+correct_count['neg']+correct_count['neu']
        num_correct_aspect = rel_count['pos']+rel_count['neg']+rel_count['neu']
        num_total = total_count['pos']+total_count['neg']+total_count['neu']

        acc_s = num_correct_overall/(num_correct_aspect+1e-6)
       
        p_pos = correct_count['pos'] / (pred_count['pos']+1e-6)
        r_pos = correct_count['pos'] / (rel_count['pos']+1e-6)
        
        p_neg = correct_count['neg'] / (pred_count['neg']+1e-6)
        r_neg = correct_count['neg'] / (rel_count['neg']+1e-6)

        p_neu = correct_count['neu'] / (pred_count['neu']+1e-6)
        r_neu= correct_count['neu'] / (rel_count['neu']+1e-6)

        pr_s = (p_pos+p_neg+p_neu)/3.0
        re_s = (r_pos+r_neg+r_neu)/3.0
        # F1 score for AS only
        f_s = 2*pr_s*re_s/(pr_s+re_s)
       
        precision_absa = num_correct_overall/(predicted+1e-6 - predicted_conf)
        recall_absa = num_correct_overall/(num_total+1e-6)
        # F1 score of the end-to-end task
        f_absa = 2*precision_absa*recall_absa/(precision_absa+recall_absa+1e-6)

    return f_aspect, acc_s, f_s, f_absa


def get_metric(y_true_aspect, y_predict_aspect, y_true_sentiment, y_predict_sentiment, mask, train_op):
    f_a, f_o = 0, 0
    true_aspect, true_sentiment = convert_to_list(y_true_aspect, y_true_sentiment, mask)
    predict_aspect, predict_sentiment = convert_to_list(y_predict_aspect, y_predict_sentiment, mask)

    f_aspect, acc_s, f_s, f_absa = score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, 0)

    if train_op:
        f_opinion, _, _, _ = score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, 1)

    return f_aspect, f_opinion, acc_s, f_s, f_absa



