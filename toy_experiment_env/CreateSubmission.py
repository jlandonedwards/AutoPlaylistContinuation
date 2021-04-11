import tensorflow as tf
import json
import csv
import time

# TODO optimize to speed up/parallelize
def get_challenge_submission(pids, rec_tracks, tid_2_uri_dict):
    
    def convert_tid_2_uri(tracks_tensor):
        return(tf.map_fn(fn = lambda x: tid_2_uri_dict[str(x.numpy())], elems = tracks_tensor, fn_output_signature = tf.string))
    
    submission_tracks = tf.map_fn(fn = lambda x: convert_tid_2_uri(x), elems = rec_tracks, fn_output_signature = tf.TensorSpec(rec_tracks[0].shape, tf.string))

    submissions = list(map(lambda x: [pids[x]] + ["spotify:track:" + a.decode("utf-8") for a in list(submission_tracks.numpy()[x])], range(0,len(pids))))

    return submissions

def process_challenge_batch_for_prediction(current_batch):
    # tracks
    list_of_track_tensors = [row[0] for row in current_batch]
    vals1 = tf.concat(list_of_track_tensors, axis = 0)
    lens1 = tf.stack([tf.shape(t, out_type = tf.int64)[0] for t in list_of_track_tensors])
    x_tracks = tf.RaggedTensor.from_row_lengths(vals1, lens1)
    del list_of_track_tensors

    # artists
    list_of_artist_tensors = [row[1] for row in current_batch]
    vals2 = tf.concat(list_of_artist_tensors, axis = 0)
    lens2 = tf.stack([tf.shape(t, out_type = tf.int64)[0] for t in list_of_artist_tensors])
    x_artists = tf.RaggedTensor.from_row_lengths(vals2, lens2)
    del list_of_artist_tensors

    list_of_pids = [row[3].numpy()[0] for row in current_batch]

    return x_tracks, x_artists, list_of_pids

def write_submission_to_file(submissions, path_to_file):
    
    with open(path_to_file, 'w', newline = '') as outFile:
        wr = csv.writer(outFile, quoting = csv.QUOTE_NONE)
        wr.writerow(['team_info'] + ['my awesome team name'] + ['my_awesome_team@email.com'])
        wr.writerows(submissions)
        outFile.close()

    
if __name__ ==  "__main__":
    from wip_DAE import *
    from DataLoader import *
    from DataPreprocess import *

    start = time.time()
    
    #####################
    # SETUP
    #####################
    
    # data = DataPreprocess('./toy_preprocessed')
    # data.process_train_val_data('./toy_data', 2, 2)
    # data.process_challenge_data('./challenge_data')
    # Run script 'trainingValidationSplit.py'
    
    BATCH_SIZE = 50
    EPOCH = 1
    
    dataset = DataLoader('./toy_preprocessed/id_dicts')
    training_set = dataset.get_traing_set('./toy_train',BATCH_SIZE,123)
    validation_sets = dataset.get_validation_sets('./toy_val')
    challenge_sets = dataset.get_challenge_sets('./toy_preprocessed/challenge_data')

    model = DAE(BATCH_SIZE)
    opt = keras.optimizers.Adam()
    
    #####################
    # TRAIN MODEL
    #####################
    
    # print("Initial Training")
        
    # count = 0
    
    # for epoch in range(EPOCH):
    #     for x_tracks,x_artists,y_tracks,y_artists in training_set:
    #         with tf.GradientTape() as tape:
    #             y_pred = model(tf.concat([x_tracks,x_artists],axis=1), training=False)  # Forward pass
    #             # Compute our own loss
    #             loss = model.loss(y_tracks,y_artists,y_pred)
    #             # Compute gradients
    #             trainable_vars = model.trainable_variables
    #             gradients = tape.gradient(loss, trainable_vars)

    #         # Update weights
    #         opt.apply_gradients(zip(gradients, trainable_vars))
            
    #         rec_tracks,rec_artists = model.get_reccomendations(x_tracks,y_tracks,y_artists,y_pred)
    #         r_precision,ndcg,rec_clicks = model.Metrics.collect_metrics(1,loss,rec_tracks,rec_artists,y_tracks,y_artists)
    #         print("[Batch #{0}],loss:{1:.2f},R-precison:{2:.2f},NDCG:{3:.2f},Rec-Clicks:{4:.2f}".format(count,loss,r_precision,ndcg,rec_clicks))
    #         count +=1
    #     model.Metrics.collect_metrics(0)
    
    # model.save_weights("weights",save_format="tf")
    # print("Done Initial training")
    
    #####################
    # LOAD TRAINED MODEL AND GENERATE SUBMISSION
    #####################
    
    reconstructed_model = DAE(BATCH_SIZE)
    reconstructed_model.load_weights("weights")

    submissions = []
    ctr = 0
    
    with open('./toy_preprocessed/challenge_data') as cfile:
        cdata = json.load(cfile)
        cfile.close()
        tid_2_uri_dict = cdata['tid_2_uri']
        del(cdata)
    
    for current_batch in challenge_sets:
        
        # TODO skipping first batch for now since tracks are nonexistent
        print(f"Batch number = {ctr}")
        if (ctr == 0):
            ctr = 1
            continue

        ctr = ctr + 1

        x_tracks, x_artists, pids = process_challenge_batch_for_prediction(current_batch = current_batch)
        
        # Since x_artists is supposedly empty, seems no need to concatenate x_tracks with x_artists
        y_pred = model(x_tracks, training=False)

        rec_tracks,rec_artists = model.get_reccomendations(x_tracks = x_tracks, y_tracks = None, y_artists = None, y_pred = y_pred)       
        
        # Takes about 5 minutes (see get_challenge_submission() definition)
        submissions = submissions + get_challenge_submission(pids, rec_tracks, tid_2_uri_dict)
        
        
    print("Submissions generated, outputting to file")

    submission_file_name = 'submission.csv'
    write_submission_to_file(submissions, submission_file_name)

    end = time.time()
    print(f"Done in = {end - start} seconds")