

def add_id_for_line(file1, file2):

    with open(file1, 'r', encoding='utf8') as f1_r, \
        open(file2, 'r', encoding='utf8') as f2_r, \
        open(file1 + '.w', 'w', encoding='utf8') as f1_w, \
        open(file2 + '.w', 'w', encoding='utf8') as f2_w:
        i = 1
        for ref, hy in zip(f1_r.readlines(), f2_r.readlines()):
            f1_w.writelines(ref.strip('\n') + '  ' + '({})'.format(i) + '\n')
            f2_w.writelines(hy.strip('\n') + '  ' + '({})'.format(i) + '\n')
            i += 1

def filter_score(file1, file2, file3, file4):

    with open(file1, 'r', encoding='utf8') as f1_r, \
        open(file2, 'r', encoding='utf8') as f2_r, \
        open(file3, 'r', encoding='utf8') as f3_r, \
        open(file4, 'r', encoding='utf8') as f4_r, \
        open(file1 + '.w', 'w', encoding='utf8') as f1_w, \
        open(file2 + '.w', 'w', encoding='utf8') as f2_w, \
        open(file3 + '.w', 'w', encoding='utf8') as f3_w, \
        open(file4 + '.w', 'w', encoding='utf8') as f4_w:
        i = 1
        for zh, en, hter, pe in zip(f1_r.readlines(), f2_r.readlines(), f3_r.readlines(), f4_r.readlines()):
            if float(hter.strip('\n')) > 0.2:
                continue
            f1_w.writelines(zh.strip('\n') + '\n')
            f2_w.writelines(en.strip('\n') + '\n')
            f3_w.writelines(hter)
            f4_w.writelines(pe)
            i += 1


if __name__ == '__main__':
    add_id_for_line('F:\\NMT_data_sets\\nmtqe\\qe_new_data\\train_aug_0506\\train_aug.en', 'F:\\NMT_data_sets\\nmtqe\\qe_new_data\\train_aug_0506\\output_dev')
    # filter_score('F:\\NMT_data_sets\\nmtqe\\qe_new_data\\train_aug\\train_aug_bpe.zh',
    #                 'F:\\NMT_data_sets\\nmtqe\\qe_new_data\\train_aug\\train_aug_bpe.en',
    #                 'F:\\NMT_data_sets\\nmtqe\\qe_new_data\\train_aug\\train_aug_bpe.hter',
    #              'F:\\NMT_data_sets\\nmtqe\\qe_new_data\\train_aug_bpe.en')