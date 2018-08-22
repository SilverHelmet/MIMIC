from train_rnn import define_simple_seg_rnn, load_argv
from keras.models import Model, load_model
from models.dataset import Dataset, print_eval
import sys




if __name__ == "__main__":
    args = 'x settings/catAtt_lstm.txt settings/helstm.txt settings/time_feature/time_feature_sum.txt settings/period/period_v19.txt @time_gate_type=ones|model_out=RNNmodels/death_helstm.model'.split(' ')
    setting = load_argv(args)
    setting['event_dim'] = 3418
    model_path = sys.argv[1]
    model = define_simple_seg_rnn(setting)
    model.load_weights(model_path)


    # debug
    data = Dataset('death_exper/death_test_1000.h5')
    data.load(True, False, True, None, setting)

    val_eval = datasets[1].eval(model, setting)
    print_eval('test', val_eval)
    