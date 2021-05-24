import os
import yaml # pip install pyyaml
import torch
import argparse
from otrans.model import Transformer, TransformerLanguageModel
from otrans.recognizer import TransformerRecognizer
from otrans.data import load_vocab, FeatureLoader
import json

def create_wavscp(wav_path):

    f_text = open(os.path.join(wav_path,"text"),"w")
    f_scp = open(os.path.join(wav_path,"wav.scp"),"w")
    utt_list = []
    for root, dirs, files in os.walk(wav_path):
        for file_name in files:
            if file_name.endswith("wav"):
                if file_name not in utt_list:
                    
                    wav_file = os.path.join(root,file_name)
                    f_text.write(file_name + " unk\n")
                    f_scp.write(file_name+" "+wav_file+"\n")
                else:
                    raise Exception("there are same name wavfile in path")
    f_text.close()
    f_scp.close()
                


def main(args):

    checkpoint = torch.load(args.load_model)

    params = checkpoint['params']
    params['data']['shuffle'] = False
    params['data']['spec_augment'] = False
    params['data']['short_first'] = False
    params['data']['batch_size'] = args.batch_size
    params['data']['test'] = args.data_path
    params['data']['vocab'] = args.vocab

    if args.rewrite_data:
        create_wavscp(args.data_path)

    model = Transformer(params['model'])
    model.load_state_dict(checkpoint['model'])
    print('Load pre-trained model from %s' % args.load_model)
    model.eval()
    
    with torch.no_grad():
        if args.ngpu > 0:
            model.cuda()

        char2unit = load_vocab(params['data']['vocab'])
        unit2char = {i:c for c, i in char2unit.items()}

        data_loader = FeatureLoader(params, 'test', is_eval=True)
        
        recognizer = TransformerRecognizer(
            model, lm=None, lm_weight=args.lm_weight, unit2char=unit2char, beam_width=args.beam_width,
            max_len=args.max_len, penalty=args.penalty, lamda=args.lamda, ngpu=args.ngpu)

        for step, (utt_id, batch) in enumerate(data_loader.loader):

            if args.ngpu > 0:
                inputs = batch['inputs'].cuda()
                inputs_length = batch['inputs_length'].cuda()
            else:
                inputs = batch['inputs']
                inputs_length = batch['inputs_length']
            # a = inputs.cuda().data.cpu().numpy()
            preds = recognizer.recognize(inputs, inputs_length)

            for i in range(len(preds)):
                print(utt_id[i] + " --> " + preds[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--load_model', type=str, default="/thspfs/home/acct-hpc/thspzh-hpc1/yuanhui/OpenTransformer/save_models/12-14-17-40-33/model.epoch.59.pt")
    parser.add_argument('-v','--vocab', type=str, default="/thspfs/home/acct-hpc/thspzh-hpc1/yuanhui/OpenTransformer/save_models/12-14-17-40-33/vocab")
    parser.add_argument('-ws', '--data_path', type=str, default='/thspfs/home/acct-hpc/thspzh-hpc1/dataset/Transformer_data/wavs/clear/1622')
    parser.add_argument('-rd', '--rewrite_data', type=bool, default=True)
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-bw', '--beam_width', type=int, default=5)
    parser.add_argument('-p', '--penalty', type=float, default=0.6)
    parser.add_argument('-ld', '--lamda', type=float, default=5)
    parser.add_argument('-lmw','--lm_weight', type=float, default=0.1)
    parser.add_argument('-ml', '--max_len', type=int, default=100)
    cmd_args = parser.parse_args()
    main(cmd_args)
