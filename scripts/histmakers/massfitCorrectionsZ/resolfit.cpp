// It can be ran in data mode -> takes the width biases per 4D bin as input and fits for c,d per eta bin 
// OR toys mode (closure test) -> generates width biases from dummy cd bias and fits for cd from them
// Authors: Cristina Alexe, Lorenzo Bianchini

#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include "TRandom3.h"
#include "TVector.h"
#include "TVectorT.h"
#include "TMath.h"
#include "TF1.h"
#include "TF2.h"
#include "TGraphErrors.h"
#include <TMatrixD.h>
#include <TMatrixDSymfwd.h>
#include <TStopwatch.h>
#include <ROOT/RVec.hxx>
#include <iostream>
#include <boost/program_options.hpp>
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMinimize.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/FCNGradientBase.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

//#include <Eigen/Core>
//#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;
using namespace ROOT;
using namespace ROOT::Minuit2;

typedef ROOT::VecOps::RVec<double> RVecD;
using ROOT::RDF::RNode; 

using namespace boost::program_options;

class TheoryFcn : public FCNGradientBase {
//class TheoryFcn : public FCNBase {

public:
  TheoryFcn(const int& debug, const int& seed, const int& bias, string fname, double maxSigmaErr)
    : errorDef_(1.0), debug_(debug), seed_(seed), bias_(bias), maxSigmaErr_(maxSigmaErr)
  {

    ran_ = new TRandom3(seed);

    // pT and eta binning
    if(bias==-1) { // data mode
      TFile* fin = TFile::Open(fname.c_str(), "READ");
      if(fin==0) {
	      cout << "No file!" << endl;
  	    return;
      }
      else {
	      cout << string(fin->GetName()) << " found!" << endl;
      }
      TH1F* h_pt_edges = (TH1F*)fin->Get("h_pt_edges");
      TH1F* h_eta_edges = (TH1F*)fin->Get("h_eta_edges");
      unsigned int pt_edges_size  = h_pt_edges->GetXaxis()->GetNbins()+1;
      unsigned int eta_edges_size = h_eta_edges->GetXaxis()->GetNbins()+1;
      
      pt_edges_.reserve( pt_edges_size );
      for(unsigned int i=0; i<h_pt_edges->GetXaxis()->GetNbins(); i++)
	      pt_edges_.push_back( h_pt_edges->GetXaxis()->GetBinLowEdge(i+1) );
      pt_edges_.push_back( h_pt_edges->GetXaxis()->GetBinUpEdge( h_pt_edges->GetXaxis()->GetNbins() ));

      eta_edges_.reserve( eta_edges_size );
      for(unsigned int i=0; i<h_eta_edges->GetXaxis()->GetNbins(); i++)
	      eta_edges_.push_back( h_eta_edges->GetXaxis()->GetBinLowEdge(i+1) );
      eta_edges_.push_back( h_eta_edges->GetXaxis()->GetBinUpEdge( h_eta_edges->GetXaxis()->GetNbins() ));
      fin->Close();
    }
    else { // toys mode
      pt_edges_  = {25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0}; 
      eta_edges_ = {-2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
	                  0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4};
    }

    // pt_edges_  = {25, 30, 35}; 
    //eta_edges_ = {-3.0, -2.5, -2.0};

    n_pt_bins_  = pt_edges_.size()-1;
    n_eta_bins_ = eta_edges_.size()-1;

    n_pars_ = 2*n_eta_bins_;
    
    for(auto p : pt_edges_) k_edges_.emplace_back(1./p);
    for(unsigned int i = 0; i < n_pt_bins_; i++) kmean_vals_.emplace_back( 0.5*(k_edges_[i]+k_edges_[i+1]) );
    kmean_val_ = 0.5*(kmean_vals_[n_pt_bins_-1] + kmean_vals_[0]);

    n_data_ = n_eta_bins_*n_eta_bins_*n_pt_bins_*n_pt_bins_;
    n_dof_ = 0;
    
    // Prepare storage for fit inputs and results
    sigmas2_.reserve(n_data_);
    sigmas2Err_.reserve(n_data_);
    resols2_.reserve(n_data_/2);
    masks_.reserve(n_data_);
    for(unsigned int idata = 0; idata<n_data_; idata++) {
      sigmas2_.push_back( 0.0 );
      sigmas2Err_.push_back( 0.0 );
      masks_.push_back(1);
    }
    for(unsigned int idata = 0; idata<n_data_/2; idata++) {
      resols2_.push_back(1.0e-04);
    }

    x_vals_ = VectorXd(n_pars_); // Holds all c,d parameters in a single vector in this order
    c_vals_ = VectorXd(n_eta_bins_);
    d_vals_ = VectorXd(n_eta_bins_);
    c_vals_prevfit_ = VectorXd(n_eta_bins_);
    d_vals_prevfit_ = VectorXd(n_eta_bins_);
    for(unsigned int i=0; i<n_eta_bins_; i++) {
      c_vals_(i) = 0.0;
      d_vals_(i) = 0.0;
      c_vals_prevfit_(i) = 0.0;
      d_vals_prevfit_(i) = 0.0;
    }
    for(unsigned int i=0; i<n_pars_; i++) {
      x_vals_(i) = 0.0;
    }

    if(bias_>0) { // toys mode only 
      // bias for c out
      for(unsigned int i=0; i<n_eta_bins_; i++) {
	      double val = ran_->Uniform(-0.01, 0.01);
	      if(bias_==2) {
	        double mid_point = double(n_eta_bins_)*0.5;
	        val = (i-mid_point)*(i-mid_point)/mid_point/mid_point*0.01;
	      }
	      c_vals_(i) = val;
	      x_vals_(i) = val;
      }
      // bias for d out
      for(unsigned int i=0; i<n_eta_bins_; i++) {
	      double val = ran_->Uniform(-0.01/kmean_val_, 0.01/kmean_val_);
	      if(bias_==2) {
	        double mid_point = double(n_eta_bins_)*0.5;
	        val = -(i-mid_point)*(i-mid_point)/mid_point/mid_point*0.01;
	      }
	      d_vals_(i) = val;
	      x_vals_(i+n_eta_bins_) = val;
      }
      n_dof_ = n_data_ - n_pars_;
    }
    else if(bias==-1) { // data mode
      // Read widths and masks per 4D bin from input file
      TFile* fin = TFile::Open(fname.c_str(), "READ");
      TH1D* h_widths = (TH1D*)fin->Get("h_widths");
      TH1D* h_masks = (TH1D*)fin->Get("h_masks");
      assert( h_widths->GetXaxis()->GetNbins() == n_data_);

      unsigned int n_unmasked_bins = 0;  
      for(unsigned int ibin=0;ibin<h_widths->GetXaxis()->GetNbins(); ibin++) {
	      sigmas2_[ibin]    = h_widths->GetBinContent(ibin+1)*h_widths->GetBinContent(ibin+1);
	      sigmas2Err_[ibin] = 2*TMath::Abs(h_widths->GetBinContent(ibin+1))*h_widths->GetBinError(ibin+1);
	      masks_[ibin]      = h_masks->GetBinContent(ibin+1);
	      if( h_widths->GetBinError(ibin+1)>maxSigmaErr_ ) masks_[ibin] = 0;
	      if( masks_[ibin]>0.5 ) n_unmasked_bins++;
      }

      n_dof_ = n_unmasked_bins - n_pars_;
      n_data_ = n_unmasked_bins;

      // cd values obtained in the previous fit
      TH1D* h_c_vals_prevfit = (TH1D*)fin->Get("h_c_vals_prevfit");
      TH1D* h_d_vals_prevfit = (TH1D*)fin->Get("h_d_vals_prevfit");

      for(unsigned int i=0; i<n_eta_bins_; i++) {
	      c_vals_prevfit_(i) = h_c_vals_prevfit->GetBinContent(i+1);
	      d_vals_prevfit_(i) = h_d_vals_prevfit->GetBinContent(i+1);
      }
      
      fin->Close();

      TFile* faux = TFile::Open("./root/coefficients2016ptfrom20forscaleptfrom20to70forres.root", "READ");
      if(faux!=0) {
	      TH1D* histobudget = (TH1D*)faux->Get("histobudget");
	      TH1D* histohitres = (TH1D*)faux->Get("histohitres");
	      for(unsigned int ieta=0; ieta<n_eta_bins_; ieta++) {
	        double eta = 0.5*(eta_edges_[ieta]+eta_edges_[ieta+1]);
 	        int eta_bin = histobudget->FindBin(eta);
	        if(eta_bin<1)
            eta_bin = 1;
	        else if(eta_bin>histobudget->GetNbinsX())
	          eta_bin = histobudget->GetNbinsX();
	        for(unsigned int ipt=0; ipt<n_pt_bins_; ipt++) {	    
	          double k = kmean_vals_[ipt];
	          double budget = histobudget->GetBinContent( eta_bin );
	          double hitres = histohitres->GetBinContent( eta_bin );
	          double budget2 = budget*budget;
	          double hitres2 = hitres*hitres;
	          double out =  budget2 + hitres2/k/k ;
	          resols2_[ieta*n_pt_bins_ + ipt] = out;
	        }
	      }
	      faux->Close();
      }      
    }

    // Transformation of external to internal parameters
    U_ = MatrixXd(n_pars_,n_pars_);
    for(unsigned int i=0; i<n_pars_; i++) {
      for(unsigned int j=0; j<n_pars_; j++) {
	      // block(c,c)
	      if(i<n_eta_bins_ && j<n_eta_bins_)
	        U_(i,j) = i==j ? 1.0 : 0.0;
	      // block(c,d)
	      else if(i<n_eta_bins_ && j>=n_eta_bins_ )
	        U_(i,j) = i==(j-n_eta_bins_) ? kmean_val_ : 0.0;
	      // block(d,d)
	      else if(i>=n_eta_bins_  && j>=n_eta_bins_ )
	        U_(i,j) = i==j ? kmean_val_ : 0.0;
	      else U_(i,j) = 0.0;
      }
    }
    //cout << U_ << endl;
    
  }
  
  ~TheoryFcn() { delete ran_;}

  // In toy mode, function to generate sigma^2 values from given cd
  void generate_data();

  // Function to set the seed value
  void set_seed(const int& seed){ ran_->SetSeed(seed);}

  // Function to get external or internal true parameter values from index
  double get_true_params(const unsigned int& i, const bool& external) {
    if(external)
      return x_vals_(i);
    else
      return (U_*x_vals_)(i);
  }

  double get_c_prevfit(const unsigned int& i){
    return c_vals_prevfit_(i);
  }
  double get_d_prevfit(const unsigned int& i){
    return d_vals_prevfit_(i);
  }

  unsigned int get_n_params(){ return n_pars_;}
  unsigned int get_n_data(){ return n_data_;}
  unsigned int get_n_dof(){ return n_dof_;} 

  double get_first_pt_edge(){ return pt_edges_.at(0) ;}
  double get_last_pt_edge(){ return pt_edges_.at(n_pt_bins_) ;} 

  double get_U(const unsigned int& i, const unsigned int& j) {
    return U_(i,j);
  }
  
  virtual double Up() const {return errorDef_;}
  virtual void SetErrorDef(double def) {errorDef_ = def;}

  // Function to be minimised from sigma^2 values, errors and cd parameters
  virtual double operator()(const vector<double>&) const;
  // Function for analytical gradient of the function to be minimised
  virtual vector<double> Gradient(const vector<double>& ) const;
  virtual bool CheckGradient() const {return true;} 

private:

  vector<double> sigmas2_;
  vector<double> sigmas2Err_;
  vector<double> resols2_;
  vector<int> masks_;
  vector<float> pt_edges_;
  vector<double> k_edges_;
  vector<double> kmean_vals_;
  VectorXd c_vals_;
  VectorXd d_vals_;
  VectorXd c_vals_prevfit_;
  VectorXd d_vals_prevfit_;
  VectorXd x_vals_;
  double kmean_val_;
  vector<float> eta_edges_;
  unsigned int n_pt_bins_;
  unsigned int n_eta_bins_;
  unsigned int n_data_;
  unsigned int n_pars_;
  unsigned int n_dof_;
  int debug_;
  int seed_;
  int bias_;
  double errorDef_;
  double maxSigmaErr_;
  MatrixXd U_;
  TRandom3* ran_;
};

// In toy mode, function to generate sigma^2 values from given cd
void TheoryFcn::generate_data() {
  double chi2_start = 0.;
  unsigned int ibin = 0;
  for(unsigned int ieta_p = 0; ieta_p<n_eta_bins_; ieta_p++) {
    for(unsigned int ipt_p = 0; ipt_p<n_pt_bins_; ipt_p++) {
      double k_p = kmean_vals_[ipt_p];
      for(unsigned int ieta_m = 0; ieta_m<n_eta_bins_; ieta_m++) {
	      for(unsigned int ipt_m = 0; ipt_m<n_pt_bins_; ipt_m++) {
	        double k_m = kmean_vals_[ipt_m];

          // Draw error on sigma^2 centered around ierr2_nom
	        double ierr2_nom = 0.02;
	        double ierr2 = ran_->Gaus(ierr2_nom,  ierr2_nom*0.05);
	        while(ierr2<=0.) 
	          ierr2 = ran_->Gaus(ierr2_nom,  ierr2_nom*0.05);
	      	  
          // Draw sigma^2 centered around isigma2_bias with width ierr2
	        double fp = resols2_[ieta_p*n_pt_bins_ + ipt_p]/(resols2_[ieta_p*n_pt_bins_ + ipt_p]+resols2_[ieta_m*n_pt_bins_ + ipt_m]);
	        double fm = 1.0 - fp;
	        double isigma2_bias = 1.0 +  fp*( c_vals_(ieta_p) + d_vals_(ieta_p)*k_p ) + fm*( c_vals_(ieta_m) + d_vals_(ieta_m)*k_m );
	        double isigma2 = ran_->Gaus(isigma2_bias, ierr2);
	  
          sigmas2_[ibin] = isigma2 ;
	        sigmas2Err_[ibin] = ierr2 ;
	        double dchi2 = (sigmas2_[ibin]-1.0)/sigmas2Err_[ibin];
	        //cout << dchi2*dchi2 << endl;
	        chi2_start += dchi2*dchi2 ;
	        ibin++;
	      }
      }
    }
  }
  cout << "[Toy mode] Initial chi2 = " << chi2_start << " / " << n_data_ << " ndof has prob " << TMath::Prob(chi2_start, n_data_ ) <<  endl;
  return;
}

// Define function to be minimised from sigma^2 values, errors and cd parameters
double TheoryFcn::operator()(const vector<double>& par) const {

  double val = 0.0;
  const unsigned int npars = par.size();

  unsigned int ibin = 0;
  for(unsigned int ieta_p = 0; ieta_p < n_eta_bins_; ieta_p++) {
    double c_p = par[ieta_p];
    double d_p = par[ieta_p+n_eta_bins_];
    for(unsigned int ipt_p = 0; ipt_p < n_pt_bins_; ipt_p++) {   
      double k_p = kmean_vals_[ipt_p];
      for(unsigned int ieta_m = 0; ieta_m < n_eta_bins_; ieta_m++) {
	      double c_m = par[ieta_m];
	      double d_m = par[ieta_m+n_eta_bins_];
	      for(unsigned int ipt_m = 0; ipt_m < n_pt_bins_; ipt_m++) {	  
	        double k_m = kmean_vals_[ipt_m];
	        double fp = resols2_[ieta_p*n_pt_bins_ + ipt_p]/(resols2_[ieta_p*n_pt_bins_ + ipt_p]+resols2_[ieta_m*n_pt_bins_ + ipt_m]);
	        double fm = 1.0 - fp;
	        double term = 1.0 +  fp*( c_p + d_p*(k_p-kmean_val_)/kmean_val_ ) + fm*( c_m + d_m*(k_m-kmean_val_)/kmean_val_ );
	        double ival = (sigmas2_[ibin] - term)/sigmas2Err_[ibin];
	        double ival2 = ival*ival;
	        if(masks_[ibin])
	          val += ival2;
	        ibin++;
	      }
      }
    }
  }
  // Minimize ( chi2/ndf - 1 )
  val /= n_dof_;  
  val -= 1.0;
  
  return val;
}

vector<double> TheoryFcn::Gradient(const vector<double> &par ) const {

  //cout << "Using gradient" << endl; 
  vector<double> grad(par.size(), 0.0);

  // Loop over all parameters
  for(unsigned int ipar = 0; ipar < par.size(); ipar++) {
    unsigned int ieta     = ipar % n_eta_bins_;
    unsigned int par_type = ipar / n_eta_bins_;    
    //cout << "ipar " << ipar << ": " << ieta << ", " << par_type << endl;
    double grad_i = 0.0;    
    unsigned int ibin = 0;

    // Loop over 4D bin terms
    for(unsigned int ieta_p = 0; ieta_p < n_eta_bins_; ieta_p++) {
      double c_p = par[ieta_p];
      double d_p = par[ieta_p+n_eta_bins_];
      for(unsigned int ipt_p = 0; ipt_p < n_pt_bins_; ipt_p++) {   
	      double k_p = kmean_vals_[ipt_p];       	
	      for(unsigned int ieta_m = 0; ieta_m < n_eta_bins_; ieta_m++) {
	        double c_m = par[ieta_m];
	        double d_m = par[ieta_m+n_eta_bins_];
	        for(unsigned int ipt_m = 0; ipt_m < n_pt_bins_; ipt_m++) {	  
	          double k_m = kmean_vals_[ipt_m];

	          double fp = resols2_[ieta_p*n_pt_bins_ + ipt_p]/(resols2_[ieta_p*n_pt_bins_ + ipt_p] + resols2_[ieta_m*n_pt_bins_ + ipt_m]);
	          double fm = 1.0 - fp;

	          double ig = 0.0;
	          if( ieta == ieta_p || ieta == ieta_m ) {
	            double term = 1.0 +  fp*( c_p + d_p*(k_p-kmean_val_)/kmean_val_ ) + fm*( c_m + d_m*(k_m-kmean_val_)/kmean_val_ );
	            ig = -2*(sigmas2_[ibin] - term)/sigmas2Err_[ibin]/sigmas2Err_[ibin];
	            double p_term = 0.;
	            double m_term = 0.;
	            if(par_type==0) {
		            p_term = fp;
		            m_term = fm;
	            }
	            else {
		            p_term = fp*(k_p-kmean_val_)/kmean_val_;
		            m_term = fm*(k_m-kmean_val_)/kmean_val_;
	            }
	            ig *= (p_term*( ieta == ieta_p ) + m_term*( ieta == ieta_m ) );
	          }
	          ig /= n_dof_;
	          //cout << "ibin " << ibin << " += " << ig << endl;
	          if(masks_[ibin])
	            grad_i += ig;
	          ibin++;
	        }
	      }
      }
    }
    //cout << "\t" << ipar << ": " << grad_i << endl;
    grad[ipar] = grad_i;
  }

  return grad; 
}


  
int main(int argc, char* argv[]) {

  TStopwatch sw;
  sw.Start();

  //ROOT::EnableImplicitMT();

  variables_map vm;
  try {
    options_description desc{"Options"};
    desc.add_options()
	    ("help,h", "Help screen")
	    ("ntoys",     value<long>()->default_value(1), "number of toys, should be 1 to use data")
	    ("tag",         value<std::string>()->default_value("closure"), "tag of input data")
	    ("run",         value<std::string>()->default_value("closure"), "run of input data")
	    ("bias",        value<int>()->default_value(0), "bias [-1 for data, >0 for toys: 1 for uniform random bias, 2 for eta dependent bias]")
	    ("maxSigmaErr", value<double>()->default_value(0.2), "max error on width to accept a data point")
	    ("infile",      value<std::string>()->default_value("massscales"), "type of input data")
	    ("seed",        value<int>()->default_value(4357), "seed");

    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);
    if (vm.count("help")) {
	    std::cout << desc << '\n';
	    return 0;
    }
    if (vm.count("ntoys"))    std::cout << "Number of toys: " << vm["ntoys"].as<long>() << '\n';
    if (vm.count("tag"))      std::cout << "Tag: " << vm["tag"].as<std::string>() << '\n';
    if (vm.count("run"))      std::cout << "Run: " << vm["run"].as<std::string>() << '\n';
  }
  catch (const error &ex) {
    std::cerr << ex.what() << '\n';
  }

  long ntoys         = vm["ntoys"].as<long>();
  std::string tag    = vm["tag"].as<std::string>();
  std::string infile = vm["infile"].as<std::string>();
  std::string run    = vm["run"].as<std::string>();
  int bias           = vm["bias"].as<int>();
  int seed           = vm["seed"].as<int>();
  double maxSigmaErr = vm["maxSigmaErr"].as<double>();
  
  TFile* fout = TFile::Open(("./resolfit_"+tag+"_"+run+".root").c_str(), "RECREATE");

  // Define TTree with fit qualities
  TTree* tree = new TTree("tree", "tree");
  double edm, fmin, prob;
  int isvalid, hasAccurateCovar, hasPosDefCovar, ndof;
  tree->Branch("edm", &edm, "edm/D");
  tree->Branch("fmin", &fmin, "fmin/D");
  tree->Branch("prob", &prob, "prob/D");
  tree->Branch("isvalid", &isvalid, "isvalid/I");
  tree->Branch("ndof", &ndof, "ndof/I");
  tree->Branch("hasAccurateCovar", &hasAccurateCovar, "hasAccurateCovar/I");
  tree->Branch("hasPosDefCovar", &hasPosDefCovar, "hasPosDefCovar/I");

  // Initialize function to be minimized
  int debug = 0;
  string infname = infile+"_"+tag+"_"+run+".root";
  TheoryFcn* fFCN = new TheoryFcn(debug, seed, bias, infname, maxSigmaErr);  
  fFCN->SetErrorDef(1.0 / fFCN->get_n_dof());
  unsigned int n_parameters = fFCN->get_n_params();
  MatrixXd U(n_parameters,n_parameters);
  for (int i=0; i<n_parameters; i++) {
    for (int j=0; j<n_parameters; j++) {
      U(i,j) = fFCN->get_U(i,j);
    }
  }
  MatrixXd Uinv = U.inverse();
  
  // Single vectors to store all cd parameters
  vector<double> tparIn0(n_parameters);  // internal, true
  vector<double> tparIn(n_parameters);   // internal, fitted
  vector<double> tparInErr(n_parameters);
  vector<double> tparOut0(n_parameters); // external, true
  vector<double> tparOut(n_parameters);  // external, fitted
  vector<double> tparOutErr(n_parameters);

  // Fit parameters branches
  for (int i=0; i<n_parameters/2; i++) {
    tree->Branch(Form("c%d",i),       &tparOut[i],    Form("c%d/D",i));
    tree->Branch(Form("c%d_true",i),  &tparOut0[i],   Form("c%d_true/D",i));
    tree->Branch(Form("c%d_err",i),   &tparOutErr[i], Form("c%d_err/D",i));
    tree->Branch(Form("c%d_in",i),    &tparIn[i],    Form("c%d_in/D",i));
    tree->Branch(Form("c%d_intrue",i),&tparIn0[i],    Form("c%d_intrue/D",i));
    tree->Branch(Form("c%d_inerr",i), &tparInErr[i], Form("c%d_inerr/D",i));
  }
  for (int i=0; i<n_parameters/2; i++) {
    tree->Branch(Form("d%d",i),        &tparOut[i+n_parameters/2],    Form("d%d/D",i));
    tree->Branch(Form("d%d_true",i),   &tparOut0[i+n_parameters/2],   Form("d%d_true/D",i));
    tree->Branch(Form("d%d_err",i),    &tparOutErr[i+n_parameters/2], Form("d%d_err/D",i));
    tree->Branch(Form("d%d_in",i),     &tparIn[i+n_parameters/2],     Form("d%d_in/D",i));
    tree->Branch(Form("d%d_intrue",i), &tparIn0[i+n_parameters/2],    Form("d%d_intrue/D",i));
    tree->Branch(Form("d%d_inerr",i),  &tparInErr[i+n_parameters/2],  Form("d%d_inerr/D",i));
  }

  // Fit parameters histograms
  TH1D* h_c_vals_nom  = new TH1D("h_c_vals_nom", "c nominal", n_parameters/2, 0, n_parameters/2);
  TH1D* h_d_vals_nom  = new TH1D("h_d_vals_nom", "d nominal", n_parameters/2, 0, n_parameters/2);
  TH1D* h_cin_vals_nom  = new TH1D("h_cin_vals_nom", "(c+d#bar{k})", n_parameters/2, 0, n_parameters/2);
  TH1D* h_din_vals_nom  = new TH1D("h_din_vals_nom", "d/#bar{k} nominal", n_parameters/2, 0, n_parameters/2);

  TH1D* h_c_vals_fit  = new TH1D("h_c_vals_fit", "#hat{c}", n_parameters/2, 0, n_parameters/2);
  TH1D* h_d_vals_fit  = new TH1D("h_d_vals_fit", "#hat{d}", n_parameters/2, 0, n_parameters/2);
  TH1D* h_cin_vals_fit  = new TH1D("h_cin_vals_fit", "(#hat{c}+#hat{d}#bar{k})", n_parameters/2, 0, n_parameters/2);
  TH1D* h_din_vals_fit  = new TH1D("h_din_vals_fit", "#hat{d}/#bar{k}", n_parameters/2, 0, n_parameters/2);

  TH1D* h_c_vals_prevfit  = new TH1D("h_c_vals_prevfit", "#hat{c}", n_parameters/2, 0, n_parameters/2);
  TH1D* h_d_vals_prevfit  = new TH1D("h_d_vals_prevfit", "#hat{d}", n_parameters/2, 0, n_parameters/2);

  // Widths from input cd
  TH2D* h_resols_nom   = new TH2D("h_resols_nom", "resols nominal; #eta bin", n_parameters/2, 0, n_parameters/2,
				       50, fFCN->get_first_pt_edge(), fFCN->get_last_pt_edge() );
  // Widths from previous fit cd
  TH2D* h_resols_fit   = new TH2D("h_resols_fit", "resols; #eta bin", n_parameters/2, 0, n_parameters/2,
				       50, fFCN->get_first_pt_edge(), fFCN->get_last_pt_edge() );

  unsigned int maxfcn(numeric_limits<unsigned int>::max());
  double tolerance(0.001);
  int verbosity = int(ntoys<2); 
  ROOT::Minuit2::MnPrint::SetGlobalLevel(verbosity);
  
  for(unsigned int itoy=0; itoy<ntoys; itoy++) {

    if(itoy%10==0) cout << "Toy " << itoy << " / " << ntoys << endl;

    //fFCN->set_seed(seed);
    if(bias>=0) fFCN->generate_data();
    
    // Define minimization parameters
    MnUserParameters upar;
    double start=0.0, par_error=0.01;
    for (int i=0; i<n_parameters/2; i++) upar.Add(Form("c%d",i), start, par_error);
    for (int i=0; i<n_parameters/2; i++) upar.Add(Form("d%d",i), start, par_error);

    // Minimize
    MnMigrad migrad(*fFCN, upar, 1);    
    if(itoy<1) cout << "\tMigrad..." << endl;
    FunctionMinimum min = migrad(maxfcn, tolerance);

    // Fit properties
    ndof = fFCN->get_n_dof();
    edm = double(min.Edm());
    fmin = double(min.Fval());
    prob = TMath::Prob((min.Fval()+1)*fFCN->get_n_dof(), fFCN->get_n_dof() );
    isvalid = int(min.IsValid());
    hasAccurateCovar = int(min.HasAccurateCovar());
    hasPosDefCovar = int(min.HasPosDefCovar());
    
    if(itoy<1) cout << "\tHesse..." << endl;
    MnHesse hesse(1);
    hesse(*fFCN, min);

    if(itoy<1) cout << "\t => final chi2/ndf: " << min.Fval()+1 << " (prob: " << TMath::Prob((min.Fval()+1)*fFCN->get_n_dof(), fFCN->get_n_dof() ) << ")" <<  endl;;
    
    // Internal covariance matrix
    MatrixXd Vin(n_parameters,n_parameters);    
    for(unsigned int i = 0 ; i<n_parameters; i++) {    
      for(unsigned int j = 0 ; j<n_parameters; j++) {
	      Vin(i,j) = i>j ?
	        min.UserState().Covariance().Data()[j+ i*(i+1)/2] :
	        min.UserState().Covariance().Data()[i+ j*(j+1)/2];
      }
    }
    // External covariance matrix
    MatrixXd Vout = Uinv*Vin*Uinv.transpose();

    // Internal fitted parameters
    VectorXd xin(n_parameters);
    VectorXd xinErr(n_parameters);     
    for(unsigned int i = 0 ; i<n_parameters; i++) {
      xin(i)    = min.UserState().Value(i) ;
      xinErr(i) = min.UserState().Error(i) ;
    }
    // External fitted parameters
    VectorXd x = Uinv*xin;
    VectorXd xErr(n_parameters);
    for(unsigned int i = 0 ; i<n_parameters; i++) {
      xErr(i) = TMath::Sqrt(Vout(i,i));
    }

    // Save width histograms for first toy / data
    if(itoy<1) {
      for(unsigned int ib = 0 ; ib<n_parameters/2; ib++) {
        Eigen::Vector2d xi;
        xi <<
	        x(ib) + fFCN->get_c_prevfit(ib),
	        x(ib + n_parameters/2) + fFCN->get_d_prevfit(ib);
        Eigen::Vector2d xnomi;
        xnomi <<
	        fFCN->get_true_params(ib, true),
	        fFCN->get_true_params(ib + n_parameters/2, true);
        for(unsigned int jb = 0 ; jb<h_resols_nom->GetYaxis()->GetNbins(); jb++) {
	        double kj = 1./h_resols_nom->GetYaxis()->GetBinCenter(jb+1);
	        Eigen::Vector2d aj;
	        aj << 1.0, kj;
	        Eigen::Matrix2d Vj;
	        Vj <<
	          Vout(ib, ib),                  Vout(ib, ib+n_parameters/2),
	          Vout(ib+n_parameters/2, ib),   Vout(ib+n_parameters/2, ib+n_parameters/2);
	        MatrixXd resolj    = aj.transpose()*xi;
	        MatrixXd resolnomj = aj.transpose()*xnomi;
	        MatrixXd Vresolj   = aj.transpose()*Vj*aj;
	        h_resols_nom->SetBinContent(ib+1, jb+1, TMath::Sqrt( 1.0 + resolnomj(0,0)) );
	        h_resols_fit->SetBinContent(ib+1, jb+1, TMath::Sqrt( 1.0 + resolj(0,0)) );
	        h_resols_fit->SetBinError(ib+1, jb+1, TMath::Sqrt(Vresolj(0,0)) );
        }      
      }
    }
    
    for(unsigned int i = 0 ; i<n_parameters; i++) {
      // Save parameter values
      tparIn[i]     = xin(i);
      tparIn0[i]    = fFCN->get_true_params(i, false) ;
      tparInErr[i]  = xinErr(i);
      tparOut[i]    = x(i);
      tparOut0[i]   = fFCN->get_true_params(i, true) ;
      tparOutErr[i] = xErr(i);
      //cout << "Param " << i << ": " << x(i) << " +/- " << xErr(i) << ". True value is " << fFCN->get_true_params(i, true) << endl;
      
      // Save cd histograms for first toy / data
      if(itoy<1) {
        int ip = i%(n_parameters/2);
        if(i<n_parameters/2) {
	        h_c_vals_fit->SetBinContent(ip+1, x(i));
	        h_c_vals_fit->SetBinError(ip+1, xErr(i));
	        h_c_vals_nom->SetBinContent(ip+1, fFCN->get_true_params(i, true));
	        h_cin_vals_fit->SetBinContent(ip+1, xin(i));
	        h_cin_vals_fit->SetBinError(ip+1, xinErr(i));
	        h_cin_vals_nom->SetBinContent(ip+1, fFCN->get_true_params(i, false));
	        h_c_vals_prevfit->SetBinContent(ip+1, fFCN->get_c_prevfit(ip) + x(i));
        }
        else {
	        h_d_vals_fit->SetBinContent(ip+1, x(i));
	        h_d_vals_fit->SetBinError(ip+1, xErr(i));
	        h_d_vals_nom->SetBinContent(ip+1, fFCN->get_true_params(i, true));
	        h_din_vals_fit->SetBinContent(ip+1, xin(i));
	        h_din_vals_fit->SetBinError(ip+1, xinErr(i));
	        h_din_vals_nom->SetBinContent(ip+1, fFCN->get_true_params(i, false));
	        h_d_vals_prevfit->SetBinContent(ip+1, fFCN->get_d_prevfit(ip) + x(i));
        }
      }
    }

    tree->Fill();

    // Save covariance and correlation matrices for first toy / data
    if(itoy<1) {
      TH2D* hcov = new TH2D(Form("hcov_%d", itoy), "", n_parameters, 0, n_parameters, n_parameters, 0, n_parameters);
      TH2D* hcor = new TH2D(Form("hcor_%d", itoy), "", n_parameters, 0, n_parameters, n_parameters, 0, n_parameters);  
      TH2D* hcovin = new TH2D(Form("hcovin_%d", itoy), "", n_parameters, 0, n_parameters, n_parameters, 0, n_parameters);
      TH2D* hcorin = new TH2D(Form("hcorin_%d", itoy), "", n_parameters, 0, n_parameters, n_parameters, 0, n_parameters);  

      for(unsigned int i = 0 ; i<n_parameters; i++) {    
        hcov->GetXaxis()->SetBinLabel(i+1, TString(upar.GetName(i).c_str()) );
        hcor->GetXaxis()->SetBinLabel(i+1, TString(upar.GetName(i).c_str()) );
        hcovin->GetXaxis()->SetBinLabel(i+1, TString(upar.GetName(i).c_str()) );
        hcorin->GetXaxis()->SetBinLabel(i+1, TString(upar.GetName(i).c_str()) );
        for(unsigned int j = 0 ; j<n_parameters; j++) {
	        double covin_ij = Vin(i,j);
	        double corin_ij = Vin(i,j)/TMath::Sqrt(Vin(i,i)*Vin(j,j)); 
	        double cov_ij = Vout(i,j);
	        double cor_ij = Vout(i,j)/TMath::Sqrt(Vout(i,i)*Vout(j,j)); 
	        hcov->GetYaxis()->SetBinLabel(j+1, TString(upar.GetName(j).c_str()) );
	        hcor->GetYaxis()->SetBinLabel(j+1, TString(upar.GetName(j).c_str()) );
	        hcovin->GetYaxis()->SetBinLabel(j+1, TString(upar.GetName(j).c_str()) );
	        hcorin->GetYaxis()->SetBinLabel(j+1, TString(upar.GetName(j).c_str()) );
	        hcovin->SetBinContent(i+1, j+1, covin_ij);
 	        hcorin->SetBinContent(i+1, j+1, corin_ij);
	        hcov->SetBinContent(i+1, j+1, cov_ij);
	        hcor->SetBinContent(i+1, j+1, cor_ij);
        }
      }
      
      hcor->SetMinimum(-1.0);
      hcor->SetMaximum(+1.0);
      hcorin->SetMinimum(-1.0);
      hcorin->SetMaximum(+1.0);
    
      fout->cd();
    
      hcor->Write();
      hcov->Write();
      hcorin->Write();
      hcovin->Write();
    }

    if(verbosity) {
      cout << "Data points: " << fFCN->get_n_data() << endl;
      cout << "Number of parameters: " << fFCN->get_n_params() << endl;
      cout << "chi2/ndf: " << min.Fval()+1 << " (prob: " << TMath::Prob((min.Fval()+1)*fFCN->get_n_dof(), fFCN->get_n_dof() ) << ")" <<  endl;;
      cout << "min is valid: " << min.IsValid() << std::endl;
      cout << "HesseFailed: " << min.HesseFailed() << std::endl;
      cout << "HasCovariance: " << min.HasCovariance() << std::endl;
      cout << "HasValidCovariance: " << min.HasValidCovariance() << std::endl;
      cout << "HasValidParameters: " << min.HasValidParameters() << std::endl;
      cout << "IsAboveMaxEdm: " << min.IsAboveMaxEdm() << std::endl;
      cout << "HasReachedCallLimit: " << min.HasReachedCallLimit() << std::endl;
      cout << "HasAccurateCovar: " << min.HasAccurateCovar() << std::endl;
      cout << "HasPosDefCovar : " << min.HasPosDefCovar() << std::endl;
      cout << "HasMadePosDefCovar : " << min.HasMadePosDefCovar() << std::endl;
    }
  }

  fout->cd();
  tree->Write();

  TH1D* hpulls = new TH1D("hpulls", "", n_parameters, 0, n_parameters);
  TH1D* hsigma = new TH1D("hsigma", "", n_parameters, 0, n_parameters);
  for (int i=0; i<n_parameters; i++) {
    TH1D* h = new TH1D(Form("h%d", i), "", 100,-3,3);
    int ip = i%(n_parameters/2);
    if(i<n_parameters/2){
      tree->Draw(Form("(c%d - c%d_true)/c%d_err>>h%d", ip, ip, ip, i), "", "");
      hpulls->GetXaxis()->SetBinLabel(i+1, Form("c%d", ip));
      h_c_vals_fit->GetXaxis()->SetBinLabel(ip+1, Form("c%d", ip));
      h_cin_vals_fit->GetXaxis()->SetBinLabel(ip+1, Form("cin%d", ip));
      h_c_vals_nom->GetXaxis()->SetBinLabel(ip+1, Form("c%d", ip));
      h_cin_vals_nom->GetXaxis()->SetBinLabel(ip+1, Form("cin%d", ip));
    }
    else {
      tree->Draw(Form("(d%d - d%d_true)/d%d_err>>h%d", ip, ip, ip, i), "", "");
      hpulls->GetXaxis()->SetBinLabel(i+1, Form("d%d", ip));
      h_d_vals_fit->GetXaxis()->SetBinLabel(ip+1, Form("d%d", ip));
      h_din_vals_fit->GetXaxis()->SetBinLabel(ip+1, Form("din%d", ip));
      h_d_vals_nom->GetXaxis()->SetBinLabel(ip+1, Form("d%d", ip));
      h_din_vals_nom->GetXaxis()->SetBinLabel(ip+1, Form("din%d", ip));
    }
    float pull_i = h->GetMean();
    float pull_i_err = 0.;
    float sigma_i = 0.;
    float sigma_i_err = 0.;
    if(h->GetEntries()>10) {
      h->Fit("gaus", "Q");
      TF1* gaus = (TF1*)h->GetFunction("gaus");
      if(gaus==0) {
	      cout << "no func" << endl;
	      continue;
      }
      pull_i = gaus->GetParameter(1);
      pull_i_err = gaus->GetParError(1);
      sigma_i = gaus->GetParameter(2);
      sigma_i_err = gaus->GetParError(2);
    }
    hpulls->SetBinContent(i+1, pull_i);
    hpulls->SetBinError(i+1, pull_i_err);
    hsigma->SetBinContent(i+1, sigma_i);
    hsigma->SetBinError(i+1, sigma_i_err);
    delete h;
  }
  hpulls->Write();
  hsigma->Write();

  h_c_vals_fit->Write();
  h_d_vals_fit->Write();
  h_cin_vals_fit->Write();
  h_din_vals_fit->Write();
  h_c_vals_nom->Write();
  h_d_vals_nom->Write();
  h_cin_vals_nom->Write();
  h_din_vals_nom->Write();
  h_c_vals_prevfit->Write();
  h_d_vals_prevfit->Write();
  h_resols_nom->Write();
  h_resols_fit->Write();
  
  sw.Stop();
  std::cout << "Real time: " << sw.RealTime() << " seconds " << "(CPU time:  " << sw.CpuTime() << " seconds)" << std::endl;
  fout->Close(); 

  return 0;
}
