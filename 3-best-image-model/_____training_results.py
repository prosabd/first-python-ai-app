import __read_data as rd
import ___display_data as dd
import ____display_sub_data as dsd

tdf = rd.get_data('train', 'pt112-cu113', 'train_samples_per_sec')

dd.show_all(tdf, 'Training', 'train_img_size').show()

dsd.show_subs(tdf, 'Training', 'train_img_size').show()