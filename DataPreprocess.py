#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:09:22 2021

@author: landon
"""

import sys
import json
import collections

VARIOUS_ARTISTS_URI = '0LyfQWJT6nXafLPZqxe9Of'


class DataPreprocess:
    """
    Convert original provided data into structured inputs for model (WIP) 
    """
    def __init__(self, in_path, out_path, min_track=1, min_artist=1):
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
        
        for file in in_path:
            f = open(file)
            file_data = f.read() 
            f.close()
            mpd_slice = json.load(file_data)
            for playlist in mpd_slice['playlist']:
                self.process_playlist(playlist)
        
        # Map tracks all unqiue tracks uris to interger ids
        o_track_counts = collections.OrderedDict(self.track_counts)
        tracks,track_counts,t_uri2id = self.create_ids(o_dict = o_track_counts, min_count = min_track, start_id = 1)
        del o_track_counts
        # Map all unqiue artists uris to ids
        del self.artist_counts[VARIOUS_ARTISTS_URI]
        o_artist_counts = collections.OrderedDict(self.artist_counts)
        artists,artist_counts,a_uri2id = self.create_ids(o_dict = o_artist_counts, min_count = min_artist, start_id = len(t_uri2id))
        del o_artist_counts
        
        # Convert artists and tracks uris to coressponding ids
        playlists = []
        for t_uris, a_uris, title in zip(self.playlist_tracks, self.playlist_artists, self.playlist_titles):
            tid = self.playlist_uri2id(t_uris,t_uri2id)
            aid = self.playlist_uri2id(a_uris,a_uri2id)
        
                
    def process_playlist(self,playlist):
        title = playlist['name']
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
    
    def playlist_uri2id(playlist_uris,uri2id):
        ids = []
        for uri in playlist_uris:
            i_d = uri2id.get(uri,-1)
            if i_d == -1: continue
            ids.append[i_d]
        return ids
        
        
        
        