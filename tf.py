def generate_train_dataset(train_dir):
    eo_dir = os.path.join(train_dir, 'eo')
    sar_dir = os.path.join(train_dir, 'sar')
    eo_files = glob.glob(os.path.join(eo_dir, '**', '*.tif'), recursive=True)
    filenames = [eo_file.split('/')[-1] for eo_file in eo_files]
    print(len(filenames))
    dataset = []
    for name in filenames:
        input_image = sar_image(os.path.join(sar_dir, name))
        real_image = eo_image(os.path.join(eo_dir, name))
        input_image, real_image = random_jitter(input_image, real_image)
        input_image, real_image = normalize(input_image, real_image)
        train_datapoint = [real_image, input_image]
        dataset.append(train_datapoint)
    return dataset

ds = generate_train_dataset('/kaggle/input/ai-spacetech-hackathon/train')

def separate_tensor(input):
    real_image = tf.squeeze(input[0])
    input_image = tf.squeeze(input[1])
    return input_image, real_image

train_samples = int(len(ds)*0.8)
train_dataset = tf.data.Dataset.from_tensor_slices(ds[:train_samples])
train_dataset = train_dataset.map(separate_tensor)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(ds[train_samples:])
test_dataset = test_dataset.map(separate_tensor)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#sample images
inp = ds[25][1]
re = ds[25][0]
