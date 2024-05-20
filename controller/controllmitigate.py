from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from ryu.ofproto import ofproto_v1_3


class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):

        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()

        self.flow_training()

        end = datetime.now()
        print("Training time: ", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
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
            hub.sleep(10)

            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):

        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        file0 = open("PredictFlowStatsfile.csv", "w")
        file0.write(
            'timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        for stat in sorted([flow for flow in body if (flow.priority == 1)], key=lambda flow:
        (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']

            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']

            elif stat.match['ip_proto'] == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']

            elif stat.match['ip_proto'] == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)

            try:
                packet_count_per_second = stat.packet_count / stat.duration_sec
                packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
            except:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0

            try:
                byte_count_per_second = stat.byte_count / stat.duration_sec
                byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0

            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                        .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                                stat.match['ip_proto'], icmp_code, icmp_type,
                                stat.duration_sec, stat.duration_nsec,
                                stat.idle_timeout, stat.hard_timeout,
                                stat.flags, stat.packet_count, stat.byte_count,
                                packet_count_per_second, packet_count_per_nsecond,
                                byte_count_per_second, byte_count_per_nsecond))

        file0.close()

    def flow_training(self):

        self.logger.info("Flow Training ...")

        flow_dataset = pd.read_csv('FlowStatsfile.csv')

        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

        X_flow = flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = flow_dataset.iloc[:, -1].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25,
                                                                                random_state=0)

        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = self.flow_model.predict(X_flow_test)

        self.logger.info("------------------------------------------------------------------------------")

        self.logger.info("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        self.logger.info("succes accuracy = {0:.2f} %".format(acc * 100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail * 100))
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype('float64')

            y_flow_pred = self.flow_model.predict(X_predict_flow)

            legitimate_trafic = 0
            ddos_trafic = 0

            for i, prediction in enumerate(y_flow_pred):
                if prediction == 0:
                    legitimate_trafic += 1
                else:
                    ddos_trafic += 1
                    victim = int(predict_flow_dataset.iloc[i, 5]) % 20
                    # Implement drop functionality here for detected attack
                    self.drop_attack_flow(predict_flow_dataset.iloc[i, :])

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_trafic / len(y_flow_pred) * 100) > 80:
                self.logger.info("legitimate traffic ...")
            else:
                self.logger.info("ddos traffic ...")
                self.logger.info("victim is host: h{}".format(victim))

            self.logger.info("------------------------------------------------------------------------------")

            # Clear the contents of the prediction file after processing
            file0 = open("PredictFlowStatsfile.csv", "w")
            file0.write(
                'timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            file0.close()

        except Exception as e:
            self.logger.error("An error occurred during flow prediction: {}".format(e))

    def drop_attack_flow(self, flow_data):
        """
        Function to drop flows detected as attacks.
        This function installs a flow-mod to drop packets for the detected attack flow.
        """
        try:
            datapath = self.datapaths.get(flow_data['datapath_id'])
            if datapath:
                ofproto = datapath.ofproto
                parser = datapath.ofproto_parser

                # Constructing match criteria
                match = parser.OFPMatch(
                    ipv4_src=flow_data['ip_src'],
                    ipv4_dst=flow_data['ip_dst'],
                    tcp_src=flow_data['tp_src'],
                    tcp_dst=flow_data['tp_dst'],
                    udp_src=flow_data['tp_src'],
                    udp_dst=flow_data['tp_dst'],
                    icmpv4_code=flow_data['icmp_code'],
                    icmpv4_type=flow_data['icmp_type'],
                    ip_proto=flow_data['ip_proto']
                )

                # Action to drop packet
                actions = []

                # Instructions to apply the drop action
                instructions = [
                    parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
                ]

                # Constructing the flow-mod message
                flow_mod = parser.OFPFlowMod(
                    datapath=datapath,
                    match=match,
                    instructions=instructions,
                    priority=ofproto_v1_3.OFP_DEFAULT_PRIORITY,  # Adjust priority as needed
                    table_id=0,  # Adjust table ID as needed
                    command=ofproto.OFPFC_ADD,  # Add a new flow entry
                    buffer_id=ofproto.OFP_NO_BUFFER,
                    out_port=ofproto.OFPP_ANY,
                    out_group=ofproto.OFPG_ANY,
                    flags=0,
                    cookie=0,
                    cookie_mask=0
                )

                # Sending the flow-mod message to the datapath
                datapath.send_msg(flow_mod)

                self.logger.info("Dropped attack flow: {}".format(flow_data))

        except Exception as e:
            self.logger.error("Error occurred while dropping attack flow: {}".format(e))


