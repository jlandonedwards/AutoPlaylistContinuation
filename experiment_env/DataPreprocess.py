#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:09:22 2021
@author: landon
Code was derived by following some of the implmentation by Hojin Yang:
https://github.com/hojinYang/spotify_recSys_challenge_2018/blob/master/utils/spotify_reader.py

"""
import time
import sys
import json
import collections
import re
import os
import argparse
import matplotlib.pyplot as plt

VARIOUS_ARTISTS_URI = '0LyfQWJT6nXafLPZqxe9Of'
chars = list('''abcdefghijklmnopqrstuvwxyz/<>+-1234567890''')
char2id = {ch: i for i, ch in enumerate(chars)}

def string_normalize(title):
        t = title.lower()
        t = re.sub(r"[.,#!$%\^\*;:{}=\_`~()@]", ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

MAX_TITLE_LEN = 25
def title2ids(title):
    ids = []
    for char in title:
        cid = char2id.get(char,-1)
        if cid != -1:
            ids.append(cid)
            if len(ids) == MAX_TITLE_LEN : break

    return ids


class DataPreprocess:
    """
    Convert original provided data into structured inputs for model (WIP)
    save_dir: STRING
            Path where processed data files will be is saved
    """
    def __init__(self,save_dir):
        self.save_dir = save_dir
        self.tid_2_aid = None
        
    def process_train_val_data(self, data_dir,min_track=2, min_artist=2):
        """
        
        Parameters
        ----------
        in_path : STRING
            Path to directory containing all original playlist data
        
        min_track: INT
            minimum # of appreance of a track over all playlists
        min_artist: INT
            minimum # of appearnces of an artist over all playlists

        Returns
        -------
        None.

        """
    
        self.playlists_titles = list()
        self.playlists_tracks = list()
        self.t_uri_2_a_uri = dict()
        self.track_counts = collections.Counter()
        self.artist_counts = collections.Counter()
        self.playlist_len_counts = collections.Counter()
        
        for file in os.listdir(data_dir):
            f = open(os.sep.join((data_dir,file)))
            file_data = f.read() 
            f.close()
            mpd_slice = json.loads(file_data)
            for playlist in mpd_slice['playlists']:
                self.process_playlist(playlist)
        
        # Map tracks all unqiue tracks uris to interger ids
        o_track_counts = collections.OrderedDict(self.track_counts.most_common())
        tracks,track_counts,t_uri2id = self.create_ids(o_track_counts, min_track, 0)
        del o_track_counts
        # Map all unqiue artists uris to ids
        del self.artist_counts[VARIOUS_ARTISTS_URI]
        o_artist_counts = collections.OrderedDict(self.artist_counts.most_common())
        artists,artist_counts,a_uri2id = self.create_ids(o_artist_counts, min_artist, len(t_uri2id))
        del o_artist_counts
        
        tid_2_aid = []
        for t_uri in t_uri2id.keys():
            a_uri = self.t_uri_2_a_uri[t_uri]
            a_id =  a_uri2id.get(a_uri,-1)
            if a_id == -1: continue
            tid_2_aid.append((t_uri2id[t_uri] , a_id))
            
        with open(args.utils_dir + '/tid_2_aid', 'w') as file:
            json.dump(tid_2_aid,file,indent="\t")
            file.close()
        
        self.t_uri2id = t_uri2id
        self.tid_2_aid = dict(tid_2_aid)
        
        
        # Convert artists and tracks uris to coressponding ids and titles to seperate set of character ids
        playlists = []
        for t_uris, title in zip(self.playlists_tracks, self.playlists_titles):
            tid = self.playlist_uri2id(t_uris,t_uri2id)
            if len(tid) == 0:
                continue
            cid = title2ids(title)
            self.playlist_len_counts[str(len(tid))] +=1
            playlists.append([tid,cid,[len(tid)]])
        
        data = dict()
        data_properties = dict()
        data_hist = dict()
        data_properties['max_title_len'] = MAX_TITLE_LEN
        data_properties['n_chars'] = len(char2id)
        data_properties['n_tracks'] = len(t_uri2id)
        data_properties['n_artists'] = len(a_uri2id)           
        data_properties['n_tracks_artists'] = len(t_uri2id) + len(a_uri2id)
        data_properties['n_playlists'] = len(playlists)
        data['playlists'] = playlists
        data_hist['playlists_counts'] = self.playlist_len_counts.most_common()

        
        with open(self.save_dir+'/'+'data', 'w') as file:
            json.dump(data,file,indent="\t")
            file.close()
       
        with open(args.utils_dir + '/data_properties', 'w') as file:
            json.dump(data_properties,file,indent="\t")
            file.close()
        
        with open(args.utils_dir + '/data_counts', 'w') as file:
            json.dump(data_hist,file,indent="\t")
            file.close()
        
        del data
        
        print("num playlists: %d \nnum tracks>=min_count: %d \nnum artists>=min_count: %d \nnum track and artists ids: %d" %
              (len(playlists), len(t_uri2id),len(a_uri2id),data_properties['n_tracks_artists']))
        #print(self.playlist_len_counts.most_common())
            
    def process_challenge_data(self,challenge_dir):
        if self.tid_2_aid is None:
            print("Run process_train_val_data() before processing challenge data")
            return
        
        self.playlists_titles = list()
        self.playlists_tracks = list()
        self.playlists_artists = list()
        self.count = 0
        self.id_count = 0
        challenge_playlists = []
        for file in os.listdir(challenge_dir):
            f = open(os.sep.join((challenge_dir,file)))
            file_data = f.read() 
            f.close()
            mpd_slice = json.loads(file_data)
            for playlist in mpd_slice['playlists']:
                challenge_playlists.append(self.process_challenge_playlist(playlist))
        
        data = {'challenge_playlists':challenge_playlists}
        tid_2_uri = {v: k for k, v in self.t_uri2id.items()}
        with open(self.save_dir+'/'+'challenge_data', 'w') as file:
            json.dump(data,file,indent="\t")
            file.close()
        
        with open(args.utils_dir + '/tid_2_uri', 'w') as file:
            json.dump(tid_2_uri,file,indent="\t")
            file.close()
        
    def process_playlist(self,playlist):
        title = string_normalize(playlist['name'])
        self.playlists_titles.append(title)
        tracks = []
        for track in playlist['tracks']:
            t_uri = track['track_uri'].split(':')[2]
            a_uri = track['artist_uri'].split(':')[2]
            if t_uri not in self.t_uri_2_a_uri:
                self.t_uri_2_a_uri[t_uri] = a_uri
            tracks.append(t_uri)
            self.track_counts[t_uri] += 1
            self.artist_counts[a_uri] += 1  
        self.playlists_tracks.append(tracks)
        
    def process_challenge_playlist(self,playlist):
        c_ids = []
        if 'name' in playlist:
            c_ids = title2ids(string_normalize(playlist['name']))
        t_ids = []
        a_ids = []
        pid = playlist['pid']
        for track in playlist['tracks']:
            self.count += 1
            t_uri = track['track_uri'].split(':')[2]
            t_id = self.t_uri2id.get(t_uri,-1)
            if t_id != -1:
                self.id_count += 1
                t_ids.append(t_id)
                a_id = self.tid_2_aid.get(t_id,-1)
                if a_id != -1:
                    a_ids.append(a_id)     
        return (t_ids,a_ids,c_ids,[pid])    
    
    @staticmethod   
    def create_ids(o_dict,min_count,start_id):
        uri_list = list(o_dict.keys())
        valid_uri_list = uri_list[:]
        count_list = list(o_dict.values())
        if min_count > 1:
            rm_from = count_list.index(min_count-1)
            del count_list[rm_from:]
            del valid_uri_list[rm_from:]
        uri2id = dict(zip(valid_uri_list, range(start_id, start_id + len(valid_uri_list))))
        return valid_uri_list, count_list, uri2id
    
    @staticmethod
    def playlist_uri2id(playlist_uris,uri2id):
        ids = []
        for uri in playlist_uris:
            i_d = uri2id.get(uri,-1)
            if i_d == -1: continue
            ids.append(i_d)
        return ids
        
if __name__ == '__main__':
    start_time = time.time()
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--data_dir', type=str, default='./data', help="directory where mpd slices are stored")
    args.add_argument('--challenge_data_dir', type=str, default='./challenge_data', help="directory where challenge mpd slices are stored")
    args.add_argument('--save_dir', type=str, default='./preprocessed_data', help="directory where to store outputed data file")
    args.add_argument('--utils_dir', type=str, default='./utils', help="directory where to store outputed data file")
    args.add_argument('--min_track', type=int, default=2, help='minimum count of tracks')
    args.add_argument('--min_artist', type=int, default=2, help='minimum count of artists')
    args = args.parse_args()
    
    if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
    if not os.path.isdir("./utils"):
        os.mkdir("./utils")
    
    dataset = DataPreprocess(args.save_dir)
    dataset.process_train_val_data(args.data_dir,args.min_track,args.min_artist)
    dataset.process_challenge_data(args.challenge_data_dir)
    
    labels, values = zip(*dataset.playlist_len_counts.most_common())
    plt.bar([int(i) for i in labels], values)
    plt.title("Playlists Length Distribution")
    plt.xlabel("Playlist Length")
    plt.ylabel("Count")
    plt.show()
    
    
            
    print("---completed in %s minutes ---" % round((time.time() - start_time)/60,2))
    



   
        
        