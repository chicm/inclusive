
from sklearn.utils import shuffle
from utils import get_tuning_meta, get_classes, get_train_val_meta

class BalancedSammpler():
    def __init__(self, meta, classes, stoi, balanced=True, min_label_num=500, max_label_num=800):
        self.meta = meta
        self.classes = classes
        self.class_set = set(classes)
        self.stoi = stoi
        self.num = meta.shape[0]
        self.min_label_num = min_label_num
        self.max_label_num = max_label_num
        self.max_empty_image_num = len(classes) * 10
        self.empty_image_num = 0

        self.n_classes = len(classes)
        self.class_counts = [0] * self.n_classes
        self.img_ids = []
        self.full_classes = set()

        if balanced:
            self.add_images_balanced()
        else:
            self.add_filtered_images()
        #print(self.class_counts)
    def add_filtered_images(self):
        meta = shuffle(self.meta)
        for row in meta.values:
            if len(self.get_label_indices(row[1])) > 0:
                self.img_ids.append(row[0])

    def get_label_indices(self, label_names):
        return [self.stoi[x] for x in label_names.strip().split() if x in self.class_set]
        #target = [ (1 if i in label_idx else 0) for i in range(len(self.classes))]

    def check_max_label_counts(self, label_indicies, rounds):
        '''
        label_indices: all label indicies for one image
        return True: the image can be added
        '''

        for label_index in label_indicies:
            #if rounds > 1 and  self.class_counts[label_index] < self.min_label_num:
            #    return True
            if self.class_counts[label_index] >= self.max_label_num:
                return False
        return True

    def add_images_balanced(self, max_rounds = 20):
        rounds = 0
        #print('sampling...')
        while len(self.full_classes) < self.n_classes:
            #print('rounds:', rounds)
            #print('.', end='')
            # row[0]: image_id, row[1]: labels
            
            #print(len(self.img_ids))
            #print(self.class_counts)
            meta = shuffle(self.meta)
            last_image_nums = len(self.img_ids)
            for row in meta.values:
                label_indicies = self.get_label_indices(row[1])
                #print(label_indicies)
                # check max value
                if not self.check_max_label_counts(label_indicies, rounds):
                    continue
                # add image:
                for label_index in label_indicies:
                    self.class_counts[label_index] += 1
                    if self.class_counts[label_index] >= self.min_label_num:
                        self.full_classes.add(label_index)

                if len(label_indicies) > 0:
                    self.img_ids.append(row[0])
                elif self.empty_image_num < self.max_empty_image_num:
                    self.empty_image_num += 1
                    self.img_ids.append(row[0]) 
                else:
                    pass

                if len(self.full_classes) >=  self.n_classes:
                    break
            if len(self.img_ids) == last_image_nums:
                break
            rounds += 1
            if rounds > max_rounds:
                break

if __name__ == '__main__':
    classes, stoi = get_classes('trainable', 0, 500)
    meta, _ = get_train_val_meta('trainable', 0, 500)
    #print(meta.head())
    #print(meta.shape)
    
    sampler = BalancedSammpler(meta, classes, stoi, balanced=True)

    #print(len(sampler.img_ids))
    print(sampler.img_ids[:10])
    print(sampler.class_counts)
    
    df1 = meta.set_index('ImageID')
    print(df1.head())
    selected = df1.loc[sampler.img_ids]
    
    print('selected:')
    print(selected.shape)
    print(selected.head())
    #print(meta.loc['4f4a02a53103537e'])