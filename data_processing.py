import random, math, os
from itertools import cycle

def load_pairs(data_dir):
    diff_train_pairs = []
    same_train_pairs = []
    diff_val_pairs = []
    same_val_pairs = []
    diff_test_pairs = []
    same_test_pairs = []
    
    # Load different label pairs
    for folder in os.listdir(data_dir):
        if folder.endswith("_forg"): 
            continue

        real_folder = os.path.join(data_dir, folder)
        forg_folder = os.path.join(data_dir, folder + "_forg")

        if not os.path.exists(forg_folder):
            print(f"Skipping {folder}, as no corresponding forgery folder was found.")
            continue

        real_images = sorted(os.listdir(real_folder))
        forged_images = sorted(os.listdir(forg_folder))
        
        forged_images_cycle = cycle(forged_images)

        for real_img in real_images:
            real_img_path = os.path.join(real_folder, real_img)
            # for _ in range(num_pairs_per_label):
            forged_img_path = os.path.join(forg_folder, next(forged_images_cycle))
            diff_train_pairs.append((real_img_path, forged_img_path))
       
    # Load same label pairs 
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        images = sorted(os.listdir(folder_path))
        if len(images) < 2:
            print(f"Skipping folder {folder}, as it has less than 2 images.")
            continue
        
        for _ in range(9): # Calculated based on number of image pairs in real and forged folders
            img1, img2 = random.sample(images, 2)
            img1_path = os.path.join(folder_path, img1)
            img2_path = os.path.join(folder_path, img2)
            same_train_pairs.append((img1_path, img2_path))
            
    random.shuffle(diff_train_pairs)
    random.shuffle(same_train_pairs)
    
    split_idx_diff_train = math.ceil(len(diff_train_pairs) * 0.8)
    split_idx_diff_val = math.ceil(len(diff_train_pairs) * 0.9)
    diff_train_pairs, diff_val_pairs, diff_test_pairs = diff_train_pairs[:split_idx_diff_train], diff_train_pairs[split_idx_diff_train:split_idx_diff_val], diff_train_pairs[split_idx_diff_val:]
    
    split_idx_same_train = math.ceil(len(same_train_pairs) * 0.8)
    split_idx_same_val = math.ceil(len(same_train_pairs) * 0.9)
    same_train_pairs, same_val_pairs, same_test_pairs = same_train_pairs[:split_idx_same_train], same_train_pairs[split_idx_same_train:split_idx_same_val], same_train_pairs[split_idx_same_val:]
    
    return diff_train_pairs, same_train_pairs, diff_val_pairs, same_val_pairs, diff_test_pairs, same_test_pairs

# Label different and same pairs
def label_pairs(diff_pairs, same_pairs):
    labels = [0] * len(diff_pairs) + [1] * len(same_pairs)
    all_pairs = diff_pairs + same_pairs
    images_1 = [i[0] for i in all_pairs]
    images_2 = [i[1] for i in all_pairs]

    return images_1, images_2, labels


if __name__ == '__main__':
    data_dir = './data/sign/'
    diff_train_pairs, same_train_pairs, diff_val_pairs, same_val_pairs, diff_test_pairs, same_test_pairs = load_pairs(data_dir)
    print("Number of different label training pairs:", len(diff_train_pairs))
    print("Number of same label training pairs:", len(same_train_pairs))
    print("Number of different label validation pairs:", len(diff_val_pairs))
    print("Number of same label validation pairs:", len(same_val_pairs))
    print("Number of different label test pairs:", len(diff_test_pairs))
    print("Number of same label test pairs:", len(same_test_pairs))
