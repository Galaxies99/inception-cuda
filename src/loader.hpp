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

using std :: vector;
using std :: string;

void load_weights_from_json(const char *filename, bool debug = false) {
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
            std :: cerr << "  Layer " << i + 1 << " name: " << layer_info[i] << ", total " << weights[i].size() << " sublayers.\n";
            for (int j = 0; j < weights_info[i].size(); ++ j) 
                std :: cerr << "    Sublayer " << j + 1 << " name: " << weights_info[i][j] << ", length " << weights[i][j].size() << ".\n";
            std :: cerr << '\n';
        }
    }
}

# endif