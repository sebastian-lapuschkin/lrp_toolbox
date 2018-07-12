#ifndef RELPROPOPTS_HPP_
#define RELPROPOPTS_HPP_

namespace caffe {

  class relpropopts{
  public:

	  relpropopts():codeexectype(-1),relpropformulatype(-1),
	     lrn_forward_type(-1),lrn_backward_type(-1),
	     maxpoolingtoavgpoolinginbackwardpass(0),biastreatmenttype(0),
	     epsstab(1.0),alphabeta_beta(0.1),lastlayerindex(-1),firstlayerindex(-1),
	     auxiliaryvariable_maxlayerindexforflatdistinconv(-1),has_init_relevance(false),relevance_for_init(1.0)
      {

      }

	  int codeexectype;
	  int relpropformulatype;
	  int lrn_forward_type;
	  int lrn_backward_type;
	  int maxpoolingtoavgpoolinginbackwardpass;
	  int biastreatmenttype;


	  float epsstab;
	  float alphabeta_beta;

	  int lastlayerindex;
	  int firstlayerindex;

	  std::vector<std::vector <float> > imagemeancopy;

	  int numclasses;

	  int auxiliaryvariable_maxlayerindexforflatdistinconv;

    bool has_init_relevance;
    float relevance_for_init;
  };

}

#endif
