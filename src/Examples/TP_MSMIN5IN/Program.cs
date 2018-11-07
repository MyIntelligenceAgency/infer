using System;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace TP_MSMIN5IN
{
	partial class Program
	{
		static void Main(string[] args)
		{
			//DureeCycliste1();
			//DureeCycliste2();
			DureeCycliste3();
			Console.ReadKey();
		}

		static void DureeCycliste1()
		{

			//1 - Définition du  modèle
			Variable<double> dureeMoyenne = Variable.GaussianFromMeanAndPrecision(2, 0.01);
			Variable<double> bruitTrafic = Variable.GammaFromShapeAndScale(2, 0.5);
			Variable<double> dureeLundi = Variable.GaussianFromMeanAndPrecision(dureeMoyenne, bruitTrafic);
			Variable<double> dureeMardi = Variable.GaussianFromMeanAndPrecision(dureeMoyenne, bruitTrafic);
			Variable<double> dureeMercredi = Variable.GaussianFromMeanAndPrecision(dureeMoyenne, bruitTrafic);

			// 2 - Observations et entraînement du modèle
			dureeLundi.ObservedValue = 13;
			dureeMardi.ObservedValue = 17;
			dureeMercredi.ObservedValue = 16;
			InferenceEngine engine = new InferenceEngine();
			Gaussian moyennePosterieure = engine.Infer<Gaussian>(dureeMoyenne);
			Gamma bruitPosterieur = engine.Infer<Gamma>(bruitTrafic);
			Console.WriteLine($"moyenne posterieure {moyennePosterieure.ToString()}");
			Console.WriteLine($"bruit posterieur {bruitPosterieur.ToString()}");

			// 3 - Utilisation du modèle pour faire des prédictions
			Variable<double> dureeDemain = Variable.GaussianFromMeanAndPrecision(dureeMoyenne, bruitTrafic);
			Gaussian distribDemain = engine.Infer<Gaussian>(dureeDemain);
			Console.WriteLine($"Prédiction demain {distribDemain.ToString()}, écart type: {Math.Sqrt(distribDemain.GetVariance())}");

			// 4 - Faire d'autres prédictions basées sur les premières
			double probMoinsDe18Mn = engine.Infer<Bernoulli>(dureeDemain < 18.0).GetProbTrue();
			Console.WriteLine("Probabilité que le trajet prenne moins de 18mn: {0:f2}", probMoinsDe18Mn);
		}

		static void DureeCycliste2()
		{
			var sw = Stopwatch.StartNew();
			
			// 1 - Entraînement

			double[] donneesTrajets = new[] { 13, 17, 20, 25, 16, 11, 16, 14, 12.5 };
			DonneesCycliste mesDistributions = new DonneesCycliste(
			  Gaussian.FromMeanAndPrecision(1, 0.01),
			 Gamma.FromShapeAndScale(2, 0.5));
			EntrainementCycliste monEntrainement = new EntrainementCycliste();
			monEntrainement.CreationModeleBayesien();
			monEntrainement.DefinirDistributions(mesDistributions);
			TimeSpan debut = sw.Elapsed;
			DonneesCycliste monPosterieur = monEntrainement.CalculePosterieurs(donneesTrajets);
			Console.WriteLine($"durée inférence {sw.Elapsed - debut}");
			Console.WriteLine($"moyenne posterieure {monPosterieur.DistribMoyenne.ToString()}");
			Console.WriteLine($"bruit posterieur {monPosterieur.DistribBruitTraffic.ToString()}");

			// 2 - Prédiction

			PredictionCycliste maPrediction = new PredictionCycliste();
			maPrediction.CreationModeleBayesien();
			maPrediction.DefinirDistributions(monPosterieur);
			Gaussian ditribDemain = maPrediction.EstimerTempsDemain();
			Console.WriteLine($"Prédiction demain {ditribDemain.ToString()}");
			Console.WriteLine($"Ecart type: {Math.Sqrt(ditribDemain.GetVariance())}");
			Console.WriteLine($"Probabilité durée < 13mn: " +
							  $"{maPrediction.EstimerTempsDemainInferieurA(13)}");

			// 3 - Apprentissage en ligne 

			double[] semaineSuivante = new Double[] { 18, 25, 30, 14, 11 };
			maPrediction.DefinirDistributions(monPosterieur);
			debut = sw.Elapsed;
			DonneesCycliste posterieurSemaineSuivante = monEntrainement.CalculePosterieurs(semaineSuivante);
			Console.WriteLine($"durée inférence {sw.Elapsed - debut}");
			Console.WriteLine($"moyenne posterieure semaine suivante {posterieurSemaineSuivante.DistribMoyenne.ToString()}");
			Console.WriteLine($"bruit posterieur semaine suivante {posterieurSemaineSuivante.DistribBruitTraffic.ToString()}");

			// 4 - Nouvelle prédiction 
			maPrediction.DefinirDistributions(posterieurSemaineSuivante);
			ditribDemain = maPrediction.EstimerTempsDemain();
			Console.WriteLine($"Prédiction semaine prochaine {ditribDemain.ToString()}");
			Console.WriteLine($"Ecart type: {Math.Sqrt(ditribDemain.GetVariance())}");
			Console.WriteLine($"Probabilité durée < 13mn: " +
							  $"{maPrediction.EstimerTempsDemainInferieurA(13)}");

		}

		static void DureeCycliste3()
		{
			// 1 - Entraînement

			double[] donneesTrajets = new[] { 13, 17, 20, 25, 16, 11, 16, 25, 12.5, 30 };
			DonneesCyclisteMixte mesDistribsAPriori;
			mesDistribsAPriori.DistribMoyenne = new Gaussian[]
			{
				new Gaussian(15, 100), //Ordinaire
				new Gaussian(30, 100) //Extraordinaire
			};
			mesDistribsAPriori.DistribBruitTraffic = new Gamma[]
			{
				new Gamma(2, 0.5), //O
				new Gamma(2, 0.5)  //E
			};
			mesDistribsAPriori.DistribMixe = new Dirichlet(1, 1);

			EntrainementCyclisteMixte monEntrainement = new EntrainementCyclisteMixte();
			monEntrainement.CreationModeleBayesien();
			monEntrainement.DefinirDistributions(mesDistribsAPriori);

			DonneesCyclisteMixte posterieur = monEntrainement.CalculePosterieurs(donneesTrajets);

			Console.WriteLine($"moyenne posterieure 1: {posterieur.DistribMoyenne[0].ToString()}");
			Console.WriteLine($"bruit posterieur 1: {posterieur.DistribBruitTraffic[0].ToString()}");
			Console.WriteLine($"moyenne posterieure 2: {posterieur.DistribMoyenne[1].ToString()}");
			Console.WriteLine($"bruit posterieur 2: {posterieur.DistribBruitTraffic[1].ToString()}");
			Console.WriteLine($"Coefficients du mélange: {posterieur.DistribMixe}");

			// 2 - Prédiction

			PredictionCyclisteMixte maPrediction = new PredictionCyclisteMixte();
			maPrediction.CreationModeleBayesien();
			maPrediction.DefinirDistributions(posterieur);
			Gaussian ditribDemain = maPrediction.EstimerTempsDemain();
			Console.WriteLine($"Prédiction demain {ditribDemain.ToString()}");
			Console.WriteLine($"Ecart type: {Math.Sqrt(ditribDemain.GetVariance())}");

		}
	}
}
