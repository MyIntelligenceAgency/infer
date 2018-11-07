using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;


namespace TP_MSMIN5IN
{
	public class PredictionCycliste : CyclisteBase
	{
		private Gaussian demainDistrib;
		public Variable<double> demainTemps;
		public override void CreationModeleBayesien()
		{
			base.CreationModeleBayesien();
			demainTemps = Variable.GaussianFromMeanAndPrecision(
			 Moyenne, Bruit);
		}
		public Gaussian EstimerTempsDemain()
		{
			demainDistrib = MoteurInference.Infer<Gaussian>(demainTemps);
			return demainDistrib;
		}
		public Bernoulli EstimerTempsDemainInferieurA(double duree)
		{
			return MoteurInference.Infer<Bernoulli>(demainTemps < duree);
		}
	}


}
