#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 20:10:52 2021

@author: ala
"""

import collections
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns


sns.set(font_scale=1.2)


class Explorer:

    def __init__(
            self,
            source,
            usecols=None,
            delimiter=None,
            index_col=None,
            **kwargs):

        self.source_dir = os.path.split(source)[0]

        self.source = source
        self.usecols = usecols
        self.delimiter = delimiter
        self.index_col = index_col

        # Set the kwargs as attributes.
        [self.__setattr__(k, v) for k, v in kwargs.items()]

        if (isinstance(self.source, list) or
                isinstance(self.source, tuple)):
            self.source = os.path.join(*self.source)

        print(f'Loading `{self.source}`...')
        self.data = pd.read_csv(
            self.source, delimiter=self.delimiter, nrows=self.nrows,
            usecols=self.usecols)

        # Shuffle the rows?
        if hasattr(self, 'shuffle'):
            if self.shuffle:
                self.data = self.data.sample(frac=1).reset_index(drop=True)

        # If ``usecols`` was passed, ensure the columns are in the same order.
        if self.usecols is not None:
            self.data = self.data[self.usecols]

        print(self.data.head())
        print('Shape:', self.data.shape)

    def __call__(self, profile_report='report'):
        """
        Extract various statistics and generic descriptions of the data.

        Parameters
        ----------
        profile_report_path : str
            Pathname for saving the profile report.
        """

        self.report = pd.concat([
            pd.DataFrame({'filled': self.data.notna().mean()}),
            pd.DataFrame({'dtypes': self.data.dtypes}),
            self.data.describe(
                include='all', datetime_is_numeric=True).T], axis=1)

        if isinstance(profile_report, str):
            self.profile_report = ProfileReport(self.data)
            self.profile_report.to_file(os.path.join(
                *[self.source_dir, f'{profile_report}.html']))
            self.profile_report.to_file(os.path.join(
                *[self.source_dir, f'{profile_report}.json']))

    def select_category(self, category, value):

        return self.data[self.data[category] == value]

    @staticmethod
    def bar_plot(analysis, pdfpath, bbox_to_anchor=(1, 1.17)):

        # Prepare the analysis in a format that can be read by `sns.barplot()`.
        analyses = []
        for col in analysis.columns:
            analyses += [analysis[col].reset_index(level=0).rename(
                columns={col: 'coverage'})]
            analyses[-1]['category'] = col
        analysis = pd.concat(analyses)

        ax = sns.barplot(
            data=analysis,
            x='coverage',
            y='index',
            hue='category',
            palette='Blues_d')
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Columns')
        ax.legend(
            loc='upper right', bbox_to_anchor=bbox_to_anchor, ncol=2)

        plt.savefig(
            os.path.join(*pdfpath),
            bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def find_duplicates(iterable):

        return [item for item, count in collections.Counter(
            iterable).items() if count > 1]

    def engineer(self):

        pass


class Domain(Explorer):

    def __init__(
            self,
            source=['domain', 'domain.scalar.tsv'],
            usecols=[
                'entity',
                # Intrinsic beliefs
                'campaignmonitor_result|Campaign Monitor',
                '#Campaign Monitor.domain.belief|Campaign Monitor',
                '#Quad9.domain.belief|Quad9',
                '(Registered Domain|tldextract)#(Campaign Monitor.domain.belief|Campaign Monitor)',
                '(Registered Domain|tldextract)#(Quad9.domain.belief|Quad9)',
                '#DGABelief|Generic DGA Map',
                '#DGABelief|dga_generator_bambenek-dga',
                # Intrinsic features
                '#BenignRank.domain.belief|tranco-list-top-million',
                '#BenignRank.domain.belief|alexa-top-one-million',
                '#BenignRank.domain.belief|cisco-umbrella-popularity-list',
                '#BenignRank.domain.belief|majestic-million-top-websites',
                'Top Level Domain|tldextract',
                'domain_count|Farsight DNSDB',
                'date (first seen)|Farsight DNSDB',
                'date (last seen)|Farsight DNSDB',
                'date (creation)|WHOIS',
                'date (updated)|WHOIS',
                '(Registered Domain|tldextract).(domain_count|Farsight DNSDB)',
                '(Registered Domain|tldextract).(date (first seen)|Farsight DNSDB)',
                '(Registered Domain|tldextract).(date (last seen)|Farsight DNSDB)',
                '(Registered Domain|tldextract).(date (updated)|WHOIS)',
                '(Registered Domain|tldextract)#(BenignRank.domain.belief|alexa-top-one-million)',
                '(Registered Domain|tldextract)#(BenignRank.domain.belief|cisco-umbrella-popularity-list)',
                '(Registered Domain|tldextract)#(BenignRank.domain.belief|majestic-million-top-websites)',
                # Extrinsic beliefs
                'DNSRecord (A)|Campaign Monitor',
                '(DNSRecord (A)|NSLookup).(Abuse IP DB.IP address.confidence|Abuse IP DB)',
                '(DNSRecord (A)|Campaign Monitor)#(Abuse IP DB.IP address.belief|Abuse IP DB)',
                '(DNSRecord (A)|Farsight DNSDB)#(Abuse IP DB.IP address.belief|Abuse IP DB)',
                '(DNSRecord (A)|NSLookup)#(Abuse IP DB.IP address.belief|Abuse IP DB)',
                '(DNSRecord (A)|NSLookup)#(Campaign Monitor.domain.belief|Campaign Monitor)',
                '(DNSRecord (A)|Farsight DNSDB)#(VirusTrackerInfection.ip.belief|Virustracker)',
                '(DNSRecord (A)|NSLookup)#(VirusTrackerInfection.ip.belief|Virustracker)',
                '(DNSRecord (A)|Farsight DNSDB).(Abuse IP DB.IP address.confidence|Abuse IP DB)',
                # Extrinsic features
                '(Registered Domain|tldextract).(DNSRecord (A)|Campaign Monitor)',
                '(Registered Domain|tldextract).(DNSRecord (A)|Farsight DNSDB)',
                '(Registered Domain|tldextract).(DNSRecord (MX)|Farsight DNSDB)',
                '(Registered Domain|tldextract).(DNSRecord (NS)|Farsight DNSDB)',
                '(Registered Domain|tldextract).(DNSRecord (SOA)|Farsight DNSDB)',
                '(Registered Domain|tldextract).(DNSRecord (A)|NSLookup)',
                '(Registered Domain|tldextract).(registrant_email|WHOIS)',
                '(Registered Domain|tldextract).(registrant_name|WHOIS)',
                'a_records_status|NSLookup',
                '(DNSRecord (A)|NSLookup).(date (last seen)|Campaign Monitor)',
                '(DNSRecord (A)|Campaign Monitor).(ASN|MaxMind GeoIP)',
                '(DNSRecord (A)|Campaign Monitor).(Country Code (ISO)|MaxMind GeoIP)',
                '(DNSRecord (A)|Farsight DNSDB).(Country Code (ISO)|Abuse IP DB)',
                '(DNSRecord (A)|Farsight DNSDB).(ASN|MaxMind GeoIP)',
                '(DNSRecord (A)|Farsight DNSDB).(Country Code (ISO)|MaxMind GeoIP)',
                '(DNSRecord (A)|NSLookup).(Country Code (ISO)|Abuse IP DB)',
                '(DNSRecord (A)|NSLookup).(ASN|MaxMind GeoIP)',
                'DNSRecord (MX)|NSLookup',
                'DNSRecord (MX)|Farsight DNSDB',
                'DNSRecord (NS)|Farsight DNSDB',
                'DNSRecord (SOA)|Farsight DNSDB',
                'DNSRecord (TXT)|Farsight DNSDB',
                'DNSRecord (NS)|NSLookup',
                'DNSRecord (TXT)|NSLookup',
                'DNSRecord (CNAME)|Farsight DNSDB',
                'cname_records_status|NSLookup',
                'mx_records_status|NSLookup',
                'ns_records_status|NSLookup',
                'DNS record(SOA) Lookup Status|NSLookup',
                'txt_records_status|NSLookup'],
            delimiter='\t',
            index_col='entity',
            **kwargs):

        assert len(usecols) == len(set(usecols))

        super().__init__(
            source=source, usecols=usecols, delimiter=delimiter,
            index_col=index_col, **kwargs)

    def engineer(self):

        # Features to be set to False if ``null`` and True otherwise.
        nan_is_False = [
            '#Campaign Monitor.domain.belief|Campaign Monitor',
            '(Registered Domain|tldextract)#(Campaign Monitor.domain.belief|Campaign Monitor)',
            '(Registered Domain|tldextract)#(Quad9.domain.belief|Quad9)',
            '#DGABelief|Generic DGA Map',
            '#DGABelief|dga_generator_bambenek-dga',
        ]
        self.data[nan_is_False] = self.data[nan_is_False].notna()

        # Features to be converted to datetimes.
        str_to_dt = [
            'date (first seen)|Farsight DNSDB',
            'date (last seen)|Farsight DNSDB',
            'date (creation)|WHOIS',
            'date (updated)|WHOIS',
        ]
        for feature in str_to_dt:
            self.data[feature] = pd.to_datetime(self.data[feature])
