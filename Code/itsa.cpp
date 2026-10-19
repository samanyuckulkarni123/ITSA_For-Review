// ITSA optimized three-term domain scorer.
// Compile: c++ -O3 -std=c++17 itsa.cpp -o itsa
// Run: ./itsa ../Data/scoring_input.tsv pair_scores.csv
// The embedded coefficients are the final all-data fit selected by nested
// held-out-superfamily evaluation. Use on external data; benchmark performance
// must be reported from saved outer-fold predictions, not this all-data fit.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Token { char primary; char ss; std::string extras; };
struct Domain { std::string sid, family, pdb, sccs; std::vector<Token> tokens; };

static std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result; std::stringstream stream(text); std::string item;
    while (std::getline(stream, item, delimiter)) result.push_back(item);
    return result;
}

static double base_score(char a, char b) {
    static const std::string alphabet = "HIYVPDU";
    static const double matrix[7][7] = {
        {3,0,-1,-1,0,0,-1},{0,3,-1,-1,0,0,-1},{-1,-1,1,0,-1,-1,-1},
        {-1,-1,0,.5,-1,-1,-1},{0,0,-1,-1,3,0,-1},{0,0,-1,-1,0,3,-1},
        {-1,-1,-1,-1,-1,-1,0}
    };
    auto i = alphabet.find(a), j = alphabet.find(b);
    if (i == std::string::npos || j == std::string::npos) throw std::runtime_error("Unknown token");
    return matrix[i][j];
}

static double score_token(const Token& a, const Token& b) {
    double score = base_score(a.primary, b.primary);
    if (a.ss == b.ss) score += a.ss == 'H' ? .2 : a.ss == 'E' ? .8 : 0.0;
    else if ((a.ss == 'H' || a.ss == 'E') && (b.ss == 'H' || b.ss == 'E')) score -= .3;
    int overlap = 0;
    for (char c : a.extras) if (b.extras.find(c) != std::string::npos) ++overlap;
    return score + .35 * overlap;
}

static double sw(const std::vector<Token>& a, const std::vector<Token>& b) {
    std::vector<double> previous(b.size()+1, 0.0), current(b.size()+1, 0.0); double best = 0;
    for (size_t i=1; i<=a.size(); ++i) {
        current[0]=0;
        for (size_t j=1; j<=b.size(); ++j) {
            current[j]=std::max({0.0, previous[j-1]+score_token(a[i-1],b[j-1]), previous[j]-2.0, current[j-1]-2.0});
            best=std::max(best,current[j]);
        }
        std::swap(previous,current);
    }
    return best;
}

static double normalized(const std::vector<Token>& a, const std::vector<Token>& b) {
    double aa=sw(a,a), bb=sw(b,b); if (aa<=0 || bb<=0) throw std::runtime_error("zero self score");
    return std::clamp(2.0*sw(a,b)/(aa+bb),0.0,1.0);
}


static double js_ss_similarity(const std::vector<Token>& a, const std::vector<Token>& b) {
    const std::string states="HEC"; std::array<double,3> x{},y{};
    for(const auto&t:a){auto k=states.find(t.ss);x[k==std::string::npos?2:k]++;}
    for(const auto&t:b){auto k=states.find(t.ss);y[k==std::string::npos?2:k]++;}
    double sx=a.size(),sy=b.size(),div=0.0;
    for(int k=0;k<3;k++){
        double p=x[k]/sx,q=y[k]/sy,m=.5*(p+q);
        if(p>0)div+=.5*p*std::log(p/m);
        if(q>0)div+=.5*q*std::log(q/m);
    }
    return 1.0-std::sqrt(std::max(0.0,div));
}

static double optimized_three_term(double alignment_norm,double raw_geom,double js_ss) {
    constexpr double mean_norm=0.12914887423820803, scale_norm=0.04760496588813539;
    constexpr double mean_geom=0.13456103515210466, scale_geom=0.04605376785365196;
    constexpr double mean_ss=0.7671598612492043, scale_ss=0.14760530967799285;
    constexpr double beta_norm=4.635677894877834, beta_geom=-3.5024656114375743;
    constexpr double beta_ss=2.450916231976988, intercept=-2.814072582205499;
    double z=intercept+beta_norm*(alignment_norm-mean_norm)/scale_norm
      +beta_geom*(raw_geom-mean_geom)/scale_geom+beta_ss*(js_ss-mean_ss)/scale_ss;
    return 1.0/(1.0+std::exp(-std::clamp(z,-700.0,700.0)));
}

static double composition(const std::vector<Token>& a, const std::vector<Token>& b) {
    const std::string alphabet="HIYVPDU"; std::array<double,7> x{},y{};
    for(auto&t:a) x[alphabet.find(t.primary)]++; for(auto&t:b) y[alphabet.find(t.primary)]++;
    double dot=0,nx=0,ny=0; for(int i=0;i<7;i++){dot+=x[i]*y[i];nx+=x[i]*x[i];ny+=y[i]*y[i];}
    return nx>0&&ny>0?dot/std::sqrt(nx*ny):0;
}

static uint64_t stable_seed(const std::string& sid, int repetition) {
    uint64_t value=1469598103934665603ULL;
    for(unsigned char c:sid){value^=c;value*=1099511628211ULL;}
    return value ^ (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(repetition+1));
}

int main(int argc,char**argv){
    if(argc!=3) throw std::runtime_error("usage: pilot_score input.tsv output.csv");
    std::ifstream input(argv[1]); std::string line; std::getline(input,line); std::vector<Domain> domains;
    while(std::getline(input,line)){
        auto fields=split(line,'\t'); if(fields.size()!=5) throw std::runtime_error("bad input row");
        Domain d{fields[0],fields[1],fields[2],fields[3],{}};
        for(auto&encoded:split(fields[4],';')){if(encoded.size()<2)throw std::runtime_error("bad token");d.tokens.push_back({encoded[0],encoded[1],encoded.substr(2)});} domains.push_back(std::move(d));
    }
    constexpr int repeats=20; std::vector<std::vector<std::vector<Token>>> shuffled(repeats,std::vector<std::vector<Token>>(domains.size()));
    for(int r=0;r<repeats;r++)for(size_t i=0;i<domains.size();i++){shuffled[r][i]=domains[i].tokens;std::mt19937_64 rng(stable_seed(domains[i].sid,r));std::shuffle(shuffled[r][i].begin(),shuffled[r][i].end(),rng);}
    std::ofstream out(argv[2]);out<<"sid_1,family_1,pdb_1,sid_2,family_2,pdb_2,label,itsa_normalized,alignment_raw_geom,js_ss,optimized_three_term_score,sw_raw_score,sw_self1_score,sw_self2_score\n"<<std::setprecision(17);
    size_t count=0;
    for(size_t i=0;i<domains.size();i++)for(size_t j=i+1;j<domains.size();j++){
        if(domains[i].pdb==domains[j].pdb)continue;
        double raw=sw(domains[i].tokens,domains[j].tokens);
        double self1=sw(domains[i].tokens,domains[i].tokens);
        double self2=sw(domains[j].tokens,domains[j].tokens);
        double norm=std::clamp(2.0*raw/(self1+self2),0.0,1.0);
        double geom=raw/std::sqrt(self1*self2);
        double ss=js_ss_similarity(domains[i].tokens,domains[j].tokens);
        double learned=optimized_three_term(norm,geom,ss);
        out<<domains[i].sid<<','<<domains[i].family<<','<<domains[i].pdb<<','<<domains[j].sid<<','<<domains[j].family<<','<<domains[j].pdb<<','<<(domains[i].family==domains[j].family)<<','<<norm<<','<<geom<<','<<ss<<','<<learned<<','<<raw<<','<<self1<<','<<self2<<'\n';
        if(++count%1000==0)std::cerr<<"scored "<<count<<" pairs\n";
    }
}
