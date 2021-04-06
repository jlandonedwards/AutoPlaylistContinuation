#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:09:22 2021
@author: landon
Code was based on the implmentation by Hojin Yang:
https://github.com/hojinYang/spotify_recSys_challenge_2018/blob/master/utils/spotify_reader.py

"""

import sys
import json
import collections
import re
import os
import argparse

VARIOUS_ARTISTS_URI = '0LyfQWJT6nXafLPZqxe9Of'
chars = list('''abcdefghijklmnopqrstuvwxyz/<>+-1234567890''')
char2id = {ch: i+1 for i, ch in enumerate(chars)}

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
    ids = ids + [0]*(MAX_TITLE_LEN - len(ids))
    return ids


class DataPreprocess:
    """
    Convert original provided data into structured inputs for model (WIP) 
    """
    def __init__(self, in_path, out_path, min_track=2, min_artist=2):
        """
        
        Parameters
        ----------
        in_path : STRING
            Path to directory containing playlist data
        out_path : STRING
            Path where data is saved
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
        self.playlists_artists = list()
        self.track2artists = dict()
        self.track_counts = collections.Counter()
        self.artist_counts = collections.Counter()
        self.playlist_len_counts = collections.Counter()
        
        for file in os.listdir(in_path):
            f = open(os.sep.join((in_path,file)))
            file_data = f.read() 
            f.close()
            mpd_slice = json.loads(file_data)
            for playlist in mpd_slice['playlists']:
                self.process_playlist(playlist)
        
        # Map tracks all unqiue tracks uris to interger ids
        o_track_counts = collections.OrderedDict(self.track_counts.most_common())
        tracks,track_counts,t_uri2id = self.create_ids(o_track_counts, min_track, 1)
        del o_track_counts
        # Map all unqiue artists uris to ids
        del self.artist_counts[VARIOUS_ARTISTS_URI]
        o_artist_counts = collections.OrderedDict(self.artist_counts.most_common())
        artists,artist_counts,a_uri2id = self.create_ids(o_artist_counts, min_artist, len(t_uri2id))
        del o_artist_counts
        
        # Convert artists and tracks uris to coressponding ids and titles to seperate set of character ids
        playlists = []
        for t_uris, a_uris, title in zip(self.playlists_tracks, self.playlists_artists, self.playlists_titles):
            tid = self.playlist_uri2id(t_uris,t_uri2id)
            aid = self.playlist_uri2id(a_uris,a_uri2id)
            cid = title2ids(title)
            self.playlist_len_counts[str(len(tid))] +=1
            playlists.append([tid,aid,cid,[len(tid)],[len(aid)]])
        
        data = dict()
        data['max_title_len'] = MAX_TITLE_LEN
        data['char_vocab_size'] = len(char2id) + 1
        data['trkArt_vocab_size'] = len(t_uri2id) + len(a_uri2id) + 1
        data['t_uri2id'] = t_uri2id
        data['a_uri2id'] = a_uri2id
        data['playlists'] = playlists
        data['playlists_counts'] = self.playlist_len_counts.most_common()
        
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        with open(out_path+'/'+'data', 'w') as file:
            json.dump(data,file,indent="\t")
        
        print("num playlists: %d,tracks>=min_count: %d, artists>=min_count: %d,trkArt_vocabSize: %d" %
              (len(playlists), len(t_uri2id),len(a_uri2id),data['trkArt_vocab_size']))
        #print(self.playlist_len_counts.most_common())
        
    def process_playlist(self,playlist):
        title = string_normalize(playlist['name'])
        self.playlists_titles.append(title)
        tracks = []
        artists = []
        for track in playlist['tracks']:
            t_uri = track['track_uri'].split(':')[2]
            a_uri = track['artist_uri'].split(':')[2]
            if t_uri not in self.track2artists:
                self.track2artists[t_uri] = a_uri
            tracks.append(t_uri)
            artists.append(a_uri)
            self.track_counts[t_uri] += 1
            self.artist_counts[a_uri] += 1  
        self.playlists_tracks.append(tracks)
        self.playlists_artists.append(artists)
        
    
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
        return uri_list, count_list, uri2id
    
    @staticmethod
    def playlist_uri2id(playlist_uris,uri2id):
        ids = []
        for uri in playlist_uris:
            i_d = uri2id.get(uri,-1)
            if i_d == -1: continue
            ids.append(i_d)
        return ids
        
if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--in_path', type=str, default='./toy_data', help="directory where mpd slices are stored")
    args.add_argument('--out_path', type=str, default='./toy_preprocessed', help="directory where to store outputed data file")
    args.add_argument('--min_track', type=int, default=2, help='minimum count of tracks')
    args.add_argument('--min_artist', type=int, default=2, help='minimum count of artists')
    args = args.parse_args()
    
    DataPreprocess(args.in_path,args.out_path,args.min_track,args.min_artist)
    



   
        
        