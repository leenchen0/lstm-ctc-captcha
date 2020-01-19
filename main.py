import sys
from dataset import dataset

def train(data_folder):
    dataset = dataset(data_folder)
    print('Train ', data_folder)

def test(data_folder):
    ds = dataset(DATA_FOLDER, BATCH_SIZE)
    print('Test ', data_folder)

def print_usage():
    print('Commnad not found!\nrun `python main.py train train_data_folder` for training\nand `python main.py test test_data_folder` for test')

def main(argv):
    if len(argv) != 3:
        print_usage()
        exit(-1)

    folder = argv[2]
    if argv[1] == 'train':
        train(folder)
    elif argv[1] == 'test':
        test(folder)
    else:
        print_usage()
        exit(-1)

if __name__ == "__main__":
    main(sys.argv)
