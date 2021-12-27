# ifndef _LOADER_HPP_
# define _LOADER_HPP_
# include <string>
# include <vector>
# include <ctype.h>
# include <stdio.h>
# include <fstream>
# include <iostream>
# include <assert.h>
# include <string.h>
# include "inception.h"
# include "layers.h"

using std :: vector;
using std :: string;

void set_param(double *dst, vector <double> src) {
    dst = (double*) malloc (sizeof(double) * src.size());
    for (int i = 0; i < src.size(); ++ i)
        dst[i] = src[i];
}

Inception load_weights_from_json(const char *filename, bool debug = false) {
    FILE *fp;
    vector < vector < vector < double > > > weights;
    vector < vector < string > > weights_info;
    vector < string > layer_info;
    bool first_layer = true;
    bool first_weights;
    vector < double > cur_weights;
    vector < vector < double > > cur_weights_list;
    vector < string > cur_weights_info_list;

    char ch, ch_layername, ch_weights, ch_weightsname, ch_num;
    fp = fopen(filename, "r");
    while(fscanf(fp, "%c", &ch) != EOF) {
        if (ch == '"') {
            if (!first_layer) {
                weights.push_back(cur_weights_list);
                weights_info.push_back(cur_weights_info_list);
                cur_weights_list.clear();
                cur_weights_info_list.clear();
            } else first_layer = false;
            string layer_name = "";
            while(fscanf(fp, "%c", &ch_layername) != EOF && ch_layername != '"') layer_name = layer_name + ch_layername;
            layer_info.push_back(layer_name);
            if (debug) 
                std :: cerr << "Reading layer: " << layer_name << " ...\n";
            first_weights = true;
            while (fscanf(fp, "%c", &ch_weights) != EOF && ch_weights != '{');
            while (true) {
                if (!first_weights) {
                    cur_weights_list.push_back(cur_weights);
                    cur_weights.clear();
                } else {
                    while (ch_weights != '"' && fscanf(fp, "%c", &ch_weights) != EOF);
                    first_weights = false;
                }
                string weights_name = "";
                while(fscanf(fp, "%c", &ch_weightsname) != EOF && ch_weightsname != '"') weights_name = weights_name + ch_weightsname;
                cur_weights_info_list.push_back(weights_name);
                while(fscanf(fp, "%c", &ch_num) != EOF && ch_num != ':');
                bool after_point;
                while(fscanf(fp, "%c", &ch_num) != EOF && (ch_num != '"' && ch_num != '}')) {
                    if (isdigit(ch_num) || ch_num == '-') {
                        int sign = 1;
                        if (ch_num == '-') {
                            sign = -1;
                            fscanf(fp, "%c", &ch_num);
                        }
                        double cur_num = ch_num - '0', cur_base = 1.0;
                        after_point = false;
                        while(fscanf(fp, "%c", &ch_num) != EOF && (isdigit(ch_num) || ch_num == '.')) {
                            if (ch_num == '.') after_point = true;
                            else {
                                cur_num = cur_num * 10 + ch_num - '0';
                                if (after_point) cur_base *= 10;
                            }
                        }
                        cur_num = cur_num / cur_base * sign;
                        if (ch_num == 'e') {
                            int sci_num = 0;
                            bool sci_flag = true;
                            fscanf(fp, "%c", &ch_num);
                            if (ch_num == '-') sci_flag = false;
                            else if (isdigit(ch_num)) sci_num = ch_num - '0';
                            while(fscanf(fp, "%c", &ch_num) != EOF && isdigit(ch_num))
                                sci_num = sci_num * 10 + ch_num - '0';
                            double sci_base = 1.0;
                            for (int i = 0; i < sci_num; ++ i)
                                if (sci_flag) sci_base = sci_base * 10;
                                else sci_base = sci_base / 10;
                            cur_num *= sci_base;
                        }
                        cur_weights.push_back(cur_num);
                    }
                    if (ch_num == '}' || ch_num == '"') break;
                    
                } 
                if (ch_num == '}') break;
            }
            cur_weights_list.push_back(cur_weights);
            cur_weights.clear();
        }
    }
    weights.push_back(cur_weights_list);
    weights_info.push_back(cur_weights_info_list);
    cur_weights_list.clear();
    cur_weights_info_list.clear();
    fclose(fp);
    if (debug) {
        std :: cerr << "\n============ Layer Statistics ============\n";
        std :: cerr << "Total " << layer_info.size() << " layers.\n";
        for (int i = 0; i < layer_info.size(); ++ i) {
            assert(weights[i].size() == weights_info[i].size());
            std :: cerr << "  Layer " << i + 1 << " name: " << layer_info[i] << ", total " << weights[i].size() << " params.\n";
            for (int j = 0; j < weights_info[i].size(); ++ j)
                std :: cerr << "    Param " << j + 1 << " name: " << weights_info[i][j] << ", length " << weights[i][j].size() << ".\n";
            std :: cerr << '\n';
        }
    }
    InceptionLayer1params layer1_param;
    layer1_param.way1_w = weights[0][0][0];
    layer1_param.way1_b = weights[0][1][0];
    layer1_param.way2_w = weights[0][2][0];
    layer1_param.way2_b = weights[0][3][0];
    layer1_param.way3_w = weights[0][4][0];
    layer1_param.way3_b = weights[0][5][0];
    set_param(layer1_param.c_1_w, weights[0][6]);
    set_param(layer1_param.c_1_b, weights[0][7]);
    set_param(layer1_param.c_2_w, weights[0][8]);
    set_param(layer1_param.c_2_b, weights[0][9]);
    set_param(layer1_param.c_3_w, weights[0][10]);
    set_param(layer1_param.c_3_b, weights[0][11]);
    set_param(layer1_param.c_4_w, weights[0][12]);
    set_param(layer1_param.c_4_b, weights[0][13]);
    set_param(layer1_param.c_5_w, weights[0][14]);
    set_param(layer1_param.c_5_b, weights[0][15]);
    
    InceptionLayer2params layer2_1_params;
    set_param(layer2_1_params.way1_w, weights[1][0]);
    set_param(layer2_1_params.way1_b, weights[1][1]);
    set_param(layer2_1_params.way2_1_w, weights[1][2]);
    set_param(layer2_1_params.way2_1_b, weights[1][3]);
    set_param(layer2_1_params.way2_2_w, weights[1][4]);
    set_param(layer2_1_params.way2_2_b, weights[1][5]);
    set_param(layer2_1_params.way3_1_w, weights[1][6]);
    set_param(layer2_1_params.way3_1_b, weights[1][7]);
    set_param(layer2_1_params.way3_2_w, weights[1][8]);
    set_param(layer2_1_params.way3_2_b, weights[1][9]);
    set_param(layer2_1_params.way3_3_w, weights[1][10]);
    set_param(layer2_1_params.way3_3_b, weights[1][11]);
    set_param(layer2_1_params.way4_w, weights[1][12]);
    set_param(layer2_1_params.way4_b, weights[1][13]);

    InceptionLayer2params layer2_2_params;
    set_param(layer2_2_params.way1_w, weights[2][0]);
    set_param(layer2_2_params.way1_b, weights[2][1]);
    set_param(layer2_2_params.way2_1_w, weights[2][2]);
    set_param(layer2_2_params.way2_1_b, weights[2][3]);
    set_param(layer2_2_params.way2_2_w, weights[2][4]);
    set_param(layer2_2_params.way2_2_b, weights[2][5]);
    set_param(layer2_2_params.way3_1_w, weights[2][6]);
    set_param(layer2_2_params.way3_1_b, weights[2][7]);
    set_param(layer2_2_params.way3_2_w, weights[2][8]);
    set_param(layer2_2_params.way3_2_b, weights[2][9]);
    set_param(layer2_2_params.way3_3_w, weights[2][10]);
    set_param(layer2_2_params.way3_3_b, weights[2][11]);
    set_param(layer2_2_params.way4_w, weights[2][12]);
    set_param(layer2_2_params.way4_b, weights[2][13]);

    InceptionLayer2params layer2_3_params;
    set_param(layer2_3_params.way1_w, weights[3][0]);
    set_param(layer2_3_params.way1_b, weights[3][1]);
    set_param(layer2_3_params.way2_1_w, weights[3][2]);
    set_param(layer2_3_params.way2_1_b, weights[3][3]);
    set_param(layer2_3_params.way2_2_w, weights[3][4]);
    set_param(layer2_3_params.way2_2_b, weights[3][5]);
    set_param(layer2_3_params.way3_1_w, weights[3][6]);
    set_param(layer2_3_params.way3_1_b, weights[3][7]);
    set_param(layer2_3_params.way3_2_w, weights[3][8]);
    set_param(layer2_3_params.way3_2_b, weights[3][9]);
    set_param(layer2_3_params.way3_3_w, weights[3][10]);
    set_param(layer2_3_params.way3_3_b, weights[3][11]);
    set_param(layer2_3_params.way4_w, weights[3][12]);
    set_param(layer2_3_params.way4_b, weights[3][13]);

    InceptionLayer3params layer3_params;
    set_param(layer3_params.way1_w, weights[4][0]);
    set_param(layer3_params.way1_b, weights[4][1]);
    set_param(layer3_params.way2_1_w, weights[4][2]);
    set_param(layer3_params.way2_1_b, weights[4][3]);
    set_param(layer3_params.way2_2_w, weights[4][4]);
    set_param(layer3_params.way2_2_b, weights[4][5]);
    set_param(layer3_params.way2_3_w, weights[4][6]);
    set_param(layer3_params.way2_3_b, weights[4][7]);

    InceptionLayer4params layer4_1_params;
    set_param(layer4_1_params.way1_w, weights[5][0]);
    set_param(layer4_1_params.way1_b, weights[5][1]);
    set_param(layer4_1_params.way2_1_w, weights[5][2]);
    set_param(layer4_1_params.way2_1_b, weights[5][3]);
    set_param(layer4_1_params.way2_2_w, weights[5][4]);
    set_param(layer4_1_params.way2_2_b, weights[5][5]);
    set_param(layer4_1_params.way2_3_w, weights[5][6]);
    set_param(layer4_1_params.way2_3_b, weights[5][7]);
    set_param(layer4_1_params.way3_1_w, weights[5][8]);
    set_param(layer4_1_params.way3_1_b, weights[5][9]);
    set_param(layer4_1_params.way3_2_w, weights[5][10]);
    set_param(layer4_1_params.way3_2_b, weights[5][11]);
    set_param(layer4_1_params.way3_3_w, weights[5][12]);
    set_param(layer4_1_params.way3_3_b, weights[5][13]);
    set_param(layer4_1_params.way3_4_w, weights[5][14]);
    set_param(layer4_1_params.way3_4_b, weights[5][15]);
    set_param(layer4_1_params.way3_5_w, weights[5][16]);
    set_param(layer4_1_params.way3_5_b, weights[5][17]);
    set_param(layer4_1_params.way4_w, weights[5][18]);
    set_param(layer4_1_params.way4_b, weights[5][19]);

    InceptionLayer4params layer4_2_params;
    set_param(layer4_2_params.way1_w, weights[6][0]);
    set_param(layer4_2_params.way1_b, weights[6][1]);
    set_param(layer4_2_params.way2_1_w, weights[6][2]);
    set_param(layer4_2_params.way2_1_b, weights[6][3]);
    set_param(layer4_2_params.way2_2_w, weights[6][4]);
    set_param(layer4_2_params.way2_2_b, weights[6][5]);
    set_param(layer4_2_params.way2_3_w, weights[6][6]);
    set_param(layer4_2_params.way2_3_b, weights[6][7]);
    set_param(layer4_2_params.way3_1_w, weights[6][8]);
    set_param(layer4_2_params.way3_1_b, weights[6][9]);
    set_param(layer4_2_params.way3_2_w, weights[6][10]);
    set_param(layer4_2_params.way3_2_b, weights[6][11]);
    set_param(layer4_2_params.way3_3_w, weights[6][12]);
    set_param(layer4_2_params.way3_3_b, weights[6][13]);
    set_param(layer4_2_params.way3_4_w, weights[6][14]);
    set_param(layer4_2_params.way3_4_b, weights[6][15]);
    set_param(layer4_2_params.way3_5_w, weights[6][16]);
    set_param(layer4_2_params.way3_5_b, weights[6][17]);
    set_param(layer4_2_params.way4_w, weights[6][18]);
    set_param(layer4_2_params.way4_b, weights[6][19]);

    InceptionLayer4params layer4_3_params;
    set_param(layer4_3_params.way1_w, weights[7][0]);
    set_param(layer4_3_params.way1_b, weights[7][1]);
    set_param(layer4_3_params.way2_1_w, weights[7][2]);
    set_param(layer4_3_params.way2_1_b, weights[7][3]);
    set_param(layer4_3_params.way2_2_w, weights[7][4]);
    set_param(layer4_3_params.way2_2_b, weights[7][5]);
    set_param(layer4_3_params.way2_3_w, weights[7][6]);
    set_param(layer4_3_params.way2_3_b, weights[7][7]);
    set_param(layer4_3_params.way3_1_w, weights[7][8]);
    set_param(layer4_3_params.way3_1_b, weights[7][9]);
    set_param(layer4_3_params.way3_2_w, weights[7][10]);
    set_param(layer4_3_params.way3_2_b, weights[7][11]);
    set_param(layer4_3_params.way3_3_w, weights[7][12]);
    set_param(layer4_3_params.way3_3_b, weights[7][13]);
    set_param(layer4_3_params.way3_4_w, weights[7][14]);
    set_param(layer4_3_params.way3_4_b, weights[7][15]);
    set_param(layer4_3_params.way3_5_w, weights[7][16]);
    set_param(layer4_3_params.way3_5_b, weights[7][17]);
    set_param(layer4_3_params.way4_w, weights[7][18]);
    set_param(layer4_3_params.way4_b, weights[7][19]);

    InceptionLayer4params layer4_4_params;
    set_param(layer4_4_params.way1_w, weights[8][0]);
    set_param(layer4_4_params.way1_b, weights[8][1]);
    set_param(layer4_4_params.way2_1_w, weights[8][2]);
    set_param(layer4_4_params.way2_1_b, weights[8][3]);
    set_param(layer4_4_params.way2_2_w, weights[8][4]);
    set_param(layer4_4_params.way2_2_b, weights[8][5]);
    set_param(layer4_4_params.way2_3_w, weights[8][6]);
    set_param(layer4_4_params.way2_3_b, weights[8][7]);
    set_param(layer4_4_params.way3_1_w, weights[8][8]);
    set_param(layer4_4_params.way3_1_b, weights[8][9]);
    set_param(layer4_4_params.way3_2_w, weights[8][10]);
    set_param(layer4_4_params.way3_2_b, weights[8][11]);
    set_param(layer4_4_params.way3_3_w, weights[8][12]);
    set_param(layer4_4_params.way3_3_b, weights[8][13]);
    set_param(layer4_4_params.way3_4_w, weights[8][14]);
    set_param(layer4_4_params.way3_4_b, weights[8][15]);
    set_param(layer4_4_params.way3_5_w, weights[8][16]);
    set_param(layer4_4_params.way3_5_b, weights[8][17]);
    set_param(layer4_4_params.way4_w, weights[8][18]);
    set_param(layer4_4_params.way4_b, weights[8][19]);

    InceptionLayer5params layer5_params;
    set_param(layer5_params.way1_1_w, weights[9][0]);
    set_param(layer5_params.way1_1_b, weights[9][1]);
    set_param(layer5_params.way1_2_w, weights[9][2]);
    set_param(layer5_params.way1_2_b, weights[9][3]);
    set_param(layer5_params.way2_1_w, weights[9][4]);
    set_param(layer5_params.way2_1_b, weights[9][5]);
    set_param(layer5_params.way2_2_w, weights[9][6]);
    set_param(layer5_params.way2_2_b, weights[9][7]);
    set_param(layer5_params.way2_3_w, weights[9][8]);
    set_param(layer5_params.way2_3_b, weights[9][9]);
    set_param(layer5_params.way2_4_w, weights[9][10]);
    set_param(layer5_params.way2_4_b, weights[9][11]);

    InceptionLayer6params layer6_1_params;
    set_param(layer6_1_params.way1_w, weights[10][0]);
    set_param(layer6_1_params.way1_b, weights[10][1]);
    set_param(layer6_1_params.way23_1_w, weights[10][2]);
    set_param(layer6_1_params.way23_1_b, weights[10][3]);
    set_param(layer6_1_params.way2_2_w, weights[10][4]);
    set_param(layer6_1_params.way2_2_b, weights[10][5]);
    set_param(layer6_1_params.way3_2_w, weights[10][6]);
    set_param(layer6_1_params.way3_2_b, weights[10][7]);
    set_param(layer6_1_params.way45_1_w, weights[10][8]);
    set_param(layer6_1_params.way45_1_b, weights[10][9]);
    set_param(layer6_1_params.way45_2_w, weights[10][10]);
    set_param(layer6_1_params.way45_2_b, weights[10][11]);
    set_param(layer6_1_params.way4_3_w, weights[10][12]);
    set_param(layer6_1_params.way4_3_b, weights[10][13]);
    set_param(layer6_1_params.way5_3_w, weights[10][14]);
    set_param(layer6_1_params.way5_3_b, weights[10][15]);
    set_param(layer6_1_params.way6_w, weights[10][16]);
    set_param(layer6_1_params.way6_b, weights[10][17]);

    InceptionLayer6params layer6_2_params;
    set_param(layer6_2_params.way1_w, weights[11][0]);
    set_param(layer6_2_params.way1_b, weights[11][1]);
    set_param(layer6_2_params.way23_1_w, weights[11][2]);
    set_param(layer6_2_params.way23_1_b, weights[11][3]);
    set_param(layer6_2_params.way2_2_w, weights[11][4]);
    set_param(layer6_2_params.way2_2_b, weights[11][5]);
    set_param(layer6_2_params.way3_2_w, weights[11][6]);
    set_param(layer6_2_params.way3_2_b, weights[11][7]);
    set_param(layer6_2_params.way45_1_w, weights[11][8]);
    set_param(layer6_2_params.way45_1_b, weights[11][9]);
    set_param(layer6_2_params.way45_2_w, weights[11][10]);
    set_param(layer6_2_params.way45_2_b, weights[11][11]);
    set_param(layer6_2_params.way4_3_w, weights[11][12]);
    set_param(layer6_2_params.way4_3_b, weights[11][13]);
    set_param(layer6_2_params.way5_3_w, weights[11][14]);
    set_param(layer6_2_params.way5_3_b, weights[11][15]);
    set_param(layer6_2_params.way6_w, weights[11][16]);
    set_param(layer6_2_params.way6_b, weights[11][17]);
    
    InceptionOutputLayerparams outputlayer_params;
    set_param(outputlayer_params.fc_w, weights[12][0]);
    set_param(outputlayer_params.fc_b, weights[12][1]);

    InceptionParams inception_params;
    inception_params.param_l1 = layer1_param;
    inception_params.param_l2_1 = layer2_1_params;
    inception_params.param_l2_2 = layer2_2_params;
    inception_params.param_l2_3 = layer2_3_params;
    inception_params.param_l3 = layer3_params;
    inception_params.param_l4_1 = layer4_1_params;
    inception_params.param_l4_2 = layer4_2_params;
    inception_params.param_l4_3 = layer4_3_params;
    inception_params.param_l4_4 = layer4_4_params;
    inception_params.param_l5 = layer5_params;
    inception_params.param_l6_1 = layer6_1_params;
    inception_params.param_l6_2 = layer6_2_params;
    inception_params.param_output = outputlayer_params;    

    Inception net(3, 299);
    net.set_params(inception_params);
    return net;
}

# endif