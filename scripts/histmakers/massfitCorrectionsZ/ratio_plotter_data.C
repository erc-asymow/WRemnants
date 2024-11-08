void ratio_plotter_data( TString tag = "PostVFP", TString run = "Iter0", TString selection = "NONE" ) {

  TString plotname = TString("ratio_") + tag + TString("_") + run + TString("_") + selection + TString(".png");
  
  TCanvas* c = new TCanvas("c", "canvas", 600, 600);
  
  TFile* fsIter = TFile::Open("massscales_"+tag+"_"+run+".root", "READ");
  TString mc_name    = "h_smear0_bin_m";
  TString data_name  = "h_data_bin_m";
  TH2D* h2mass_mc = (TH2D*)fsIter->Get(mc_name);
  TH2D* h2mass_data = (TH2D*)fsIter->Get(data_name);

  TH1D* eta_edges = (TH1D*)fsIter->Get("h_eta_edges");
  TH1D* pt_edges = (TH1D*)fsIter->Get("h_pt_edges");

  unsigned int n_eta_bins = eta_edges->GetXaxis()->GetNbins();
  unsigned int n_pt_bins  = pt_edges->GetXaxis()->GetNbins();
  
  unsigned int ibin = 0;
  for(unsigned int ieta_p = 0; ieta_p<n_eta_bins; ieta_p++) {
    for(unsigned int ipt_p = 0; ipt_p<n_pt_bins; ipt_p++) {
      for(unsigned int ieta_m = 0; ieta_m<n_eta_bins; ieta_m++) {
	      for(unsigned int ipt_m = 0; ipt_m<n_pt_bins; ipt_m++) {
	        float etap = TMath::Abs(eta_edges->GetXaxis()->GetBinCenter(ieta_p+1));
	        float etam = TMath::Abs(eta_edges->GetXaxis()->GetBinCenter(ieta_m+1));
	        float ptp = TMath::Abs(pt_edges->GetXaxis()->GetBinCenter(ipt_p+1));
	        float ptm = TMath::Abs(pt_edges->GetXaxis()->GetBinCenter(ipt_m+1));
	        bool accept = true;
          // eta or pT selection cuts
	        if(selection=="CC")
	          accept = etap<1.5 && etam<1.5;
	        else if(selection=="FC")
	          accept = (etap<1.5 && etam>1.5) || (etap>1.5 && etam<1.5);
	        else if(selection=="FF")
	          accept = etap>1.5 && etam>1.5;
	        else if(selection=="LL")
	          accept =  (ptp<35 && ptm<35);
	        else if(selection=="LH")
	          accept =  (ptp<35 && ptm>35) || (ptp>35 && ptm<35);
	        else if(selection=="HH")
	          accept =  (ptp>35 && ptm>35);
	        if( !accept ) {
	          for(unsigned int iy=0; iy<h2mass_mc->GetYaxis()->GetNbins(); iy++) {
	            h2mass_mc->SetBinContent(ibin+1, iy+1, 0.0);
	            h2mass_data->SetBinContent(ibin+1, iy+1, 0.0);
	          }
	        }
	        ibin++;
	      }
      }
    }
  }

  TH1D* hmass_mc = h2mass_mc->ProjectionY("mc");
  TH1D* hmass_data = h2mass_data->ProjectionY("data");

  hmass_mc->Scale(hmass_data->Integral()/hmass_mc->Integral());
  hmass_data->Divide(hmass_mc);
  hmass_data->SetLineColor(kBlack);
  hmass_data->SetMaximum(1.1);
  hmass_data->SetMinimum(0.9);
  hmass_data->SetStats(0);
  hmass_data->SetTitle(tag+", "+run+", "+selection);

  c->cd();
  hmass_data->Draw("HISTE");

  c->SaveAs(plotname);
  
}
