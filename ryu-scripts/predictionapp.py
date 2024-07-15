import time
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import pandas as pd
import joblib
from ryu import cfg
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        CONF = cfg.CONF
        CONF.register_opts([
            cfg.StrOpt('model', default='default', help=('ML model to be loaded'))])
        CONF.register_opts([
            cfg.IntOpt('overwrite_interval', default=20, help=('Interval in seconds to overwrite flow stats file'))])
        CONF.register_opts([
            cfg.IntOpt('prediction_delay', default=5, help=('Delay in seconds for prediction'))])
        self.load_model(CONF.model)
        self.model_name = CONF.model

        # Initialize a timestamp to track the last overwrite time
        self.last_overwrite_time = time.time()
        # Set the overwrite interval in seconds (e.g., overwrite every x seconds)
        self.overwrite_interval = CONF.overwrite_interval
        self.prediction_delay = CONF.prediction_delay

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(self.prediction_delay)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        current_time = time.time()

        # Check if it's time to overwrite
        if current_time - self.last_overwrite_time >= self.overwrite_interval:
            self.last_overwrite_time = current_time
            self._overwrite_flow_stats_file(ev)
        else:
            self._append_flow_stats_file(ev)

    def _overwrite_flow_stats_file(self, ev):
        timestamp = datetime.now().timestamp()

        with open("./ryu/PredictFlowStatsfile.csv", "w") as file0:
            self._write_flow_stats(file0, ev, timestamp)

    def _append_flow_stats_file(self, ev):
        timestamp = datetime.now().timestamp()

        with open("./ryu/PredictFlowStatsfile.csv", "a") as file0:
            self._write_flow_stats(file0, ev, timestamp)

    def _write_flow_stats(self, file0, ev, timestamp):
        body = ev.msg.body

        for stat in sorted([flow for flow in body if flow.priority == 1],
                           key=lambda flow: (flow.match['eth_type'], flow.match['ipv4_src'],
                                             flow.match['ipv4_dst'], flow.match['ip_proto'])):

            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']

            icmp_code = stat.match.get('icmpv4_code', float('nan'))
            icmp_type = stat.match.get('icmpv4_type', float('nan'))
            tp_src = stat.match.get('tcp_src', 0)
            tp_dst = stat.match.get('tcp_dst', 0)

            flow_id = str(ip_src) + str(tp_src) + \
                str(ip_dst) + str(tp_dst) + str(ip_proto)

            try:
                packet_count_per_second = stat.packet_count / stat.duration_sec
                packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
            except:
                packet_count_per_second = float('nan')
                packet_count_per_nsecond = float('nan')

            try:
                byte_count_per_second = stat.byte_count / stat.duration_sec
                byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
            except:
                byte_count_per_second = float('nan')
                byte_count_per_nsecond = float('nan')

            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                        .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                                ip_proto, icmp_code, icmp_type,
                                stat.duration_sec, stat.duration_nsec,
                                stat.idle_timeout, stat.hard_timeout,
                                stat.flags, stat.packet_count, stat.byte_count,
                                packet_count_per_second, packet_count_per_nsecond,
                                byte_count_per_second, byte_count_per_nsecond))

    def load_model(self, model_name):
        try:
            self.flow_model = joblib.load('./models/' + model_name + '.pkl')
            self.logger.info(
                "Pre-trained model '{}' loaded successfully.".format(model_name))
        except Exception as e:
            self.logger.error(
                "Failed to load the pre-trained model '{}': {}".format(model_name, str(e)))
            return

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv(
                './ryu/PredictFlowStatsfile.csv')

            headers = ['timestamp', 'datapath_id', 'flow_id', 'ip_src', 'tp_src', 'ip_dst',
                       'tp_dst', 'ip_proto', 'icmp_code', 'icmp_type', 'flow_duration_sec',
                       'flow_duration_nsec', 'idle_timeout', 'hard_timeout', 'flags',
                       'packet_count', 'byte_count', 'packet_count_per_second',
                       'packet_count_per_nsecond', 'byte_count_per_second',
                       'byte_count_per_nsecond',]
            predict_flow_dataset.columns = headers

            # Remove dots from numerical columns
            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace(
                '.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace(
                '.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace(
                '.', '')

            # Convert all values to float64
            predict_flow_dataset = predict_flow_dataset.astype('float64')

            # imputer = SimpleImputer(strategy='constant', fill_value=0)
            # predict_flow_dataset_0 = imputer.fit_transform(
            #   predict_flow_dataset)

            # imputer_numerical = SimpleImputer(strategy='mean')
            # predict_flow_dataset_nd = imputer_numerical.fit_transform(
            #   predict_flow_dataset_0)
            # predict_flow_dataset = pd.DataFrame(
            #   predict_flow_dataset_nd, columns=predict_flow_dataset.columns)

            # Read the file containing original features information
            selected_features = []

            # dataset_name = self.model_name.split("-")[0]

            features_info_file_path = f"./datasets/features_info_{self.model_name}.txt"
            
            with open(features_info_file_path, 'r') as f:
                # Skip the first line
                next(f)
                # Read the selected features
                for line in f:
                    selected_features.append(line.strip())
            
            selected_features_df = predict_flow_dataset[selected_features]

            # For Debugging
            '''
            print('selected_features-> ', selected_features)
            print('df columns->', selected_features_df.columns)
            # Predict using the selected features
            try:
                # Check if the model has feature names stored
                if hasattr(self.flow_model, 'feature_names_in_'):
                    feature_names = self.flow_model.feature_names_in_
                elif hasattr(self.flow_model, 'feature_names'):
                    feature_names = self.flow_model.feature_names
                else:
                    # If feature names are not stored in the model, you need to have a record of them
                    # from your training script or data preprocessing step
                    feature_names = [
                        "Feature names not found in the model attributes"]

                print("Expected features:", feature_names)
            except Exception as e:
                print(f"An error occurred: {e}")
            '''
            y_flow_pred = self.flow_model.predict(selected_features_df)
            y_flow_pred = y_flow_pred.astype(int)
            normal_traffic_count = 0
            dos_traffic_count = 0
            # print(y_flow_pred)
            for i in y_flow_pred:
                if i == 0:
                    normal_traffic_count += 1
                else:
                    dos_traffic_count += 1
            self.logger.info(
                "==============================================================================")

            if (normal_traffic_count / len(y_flow_pred) * 100) > 80:
                self.logger.info(
                    "{} - Normal Traffic ...".format(datetime.now()))
            else:
                self.logger.info(
                    "{} - DOS Traffic Detected ...".format(datetime.now()))

            self.logger.info(
                "==============================================================================")

        except:
            pass
