// this version is for the non-redundant top-K feature selection
// in this version, ranked by earliness sctore, and classifying all the classes together, 
// the default classifier has  been added as the majority class
// output into the text file
#include <bits/stdc++.h>
#include<stdlib.h>
#include<stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include "math.h"
#include <algorithm>
#include <vector>
#include <string>
#include <bitset>
#include <time.h>
#include <limits>
#include "DataSetInformation.h"
#include "Euclidean.h"
#include "quickSort.h"
using namespace std; 

// structure of a feature
struct Feature
{
  //double * f; 
 // int segmentIndex;
  int instanceIndex; 
  int startPosition; 
  int length; 
  double label; 
  double threshold; 
  double recall; 
  double precision; 
  double fscore;
  double earlyFScore;
  bitset<ROWTRAINING> bmap;  // bitset does not change
};
// global variable
double training[ROWTRAINING][DIMENSION]; // training data set
double labelTraining[ROWTRAINING]={0}; // training data class labels
double testing [ROWTESTING][DIMENSION]; //  testing data set
double labelTesting[ROWTESTING]={0}; // testing data class labels
int predictedLabel[ROWTESTING]={INT_MAX}; // predicted label by the classifier
int predictedLength[ROWTESTING]={0};// predicted length by the classifier
bitset<ROWTRAINING> totalBmap; // the union bit map of a certain length of a certain class
bitset<ROWTRAINING> allLengthBmap;  // the union bit map of all length of a certain class
 vector<Feature *> finalFset;
 vector<Feature *> AllClassesFset; 


// functions in the same file
void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[], int len  );
void printData(int index);
double * getSegment(int segIndex, int InstanceIndex);
//void DisArray(int classIndex, int k, double ** DisArray, int rows, int columns);
void DisArray(int classIndex, int instanceIndex, double ** DisArray, int ** EndP, int rows, int columns, int MinK, int MaximalK);
void SaveDisArray(double ** DisArray, int rows, int columns, const char * filename, int ** EndP, const char * filename2);
//void SaveDisArray(double ** DisArray, int rows, int columns, const char * filename);
void loadDisArray(const char * fileName, double ** Data, int rows, int columns);
void loadDisArray(const char * fileName, int ** Data, int rows, int columns);
double getMean( double * arr, int len);  // segmentIndex starting from 1
double getSTD(double * arr, int len, double mean);
Feature * ThresholdLearningAll(int DisArrayIndex, int instanceIndex, int startPosition,  double m, int k, int targetClass, double ** DisA, double RecallThreshold, int ** EndP, int alpha);
//Feature * ThresholdLearningAll(int index, double m, int k, int targetClass, double ** DisA, double RecallThreshold);
double * getFeatureSegment(Feature *f);
void PrintFeature(Feature * f);
void PrintFeature(Feature * f, ofstream& ResultFile);
void PrintTotalBitMap();
void classification(vector<Feature *> &Fs, int classIndex, int k);
void classificationAllLength(vector<Feature *> &Fs, int classIndex, ofstream& resultFile);
string IntToString(int intValue);
void PrintFeatureMoreInfo(Feature * f);
void ReduceAllLength(vector<Feature *> &finalFset, vector<Feature *> &fSet );
double OneKDE(double * arr, int len, double q, double h, double constant);
//Feature * ThresholdLearningKDE(int index, int k, int targetClass, double ** DisA, double PrecisionThreshold, double RecallThreshold, double ProbalityThreshold);
double ComputeFScore(double recall, double precision);
Feature * ThresholdLearningKDE(int DisArrayIndex, int Instanceindex, int startPosition, int k, int targetClass, double ** DisA, double RecallThreshold, double ProbalityThreshold, int ** EndP, int alpha);
void classificationAllClasses(vector<Feature *> &Fs,ofstream& resultFile) ;
void classificationAllClassesWithDefault(vector<Feature *> &Fs, ofstream& resultFile);

void computeWholeSubstringArray(double * array1, int iLengthOfArray1,
                                double* array2, int iLengthOfArray2,
                                double * pMatrixOriginal,
                                double * pArraySubstring, double * pArraySubstringEndingPosition, int iLengthOfSubstringArray, int MaximalLength);

void DisArray2(int classIndex, int instanceIndex, double ** DisArray, int ** EndP, int rows, int columns, int MinK, int MaximalK) ;

void createOriginalMatrix(double * array1, int iLengthOfArray1,
                          double * array2, int iLengthOfArray2,
                          double * &pMatrix);

int main()
{  
    
	clock_t allStart, allEnd; 
	allStart=clock(); 
     // load training data
	LoadData(trainingFileName, training,labelTraining, ROWTRAINING);
	
	// load the testing data
    LoadData(testingFileName, testing,labelTesting, ROWTESTING);

    // compute the best match distance array for the target class.
    // create the space to store the disArray of length k for a certain class

	int option=1; // option =1 using thresholdAll, option=2, using the KDE cut
	int DisArrayOption=2;  // =1 , naive version =2 fast version
    int MaximalK=DIMENSION/2;  // maximal length
	int MinK=5;
	double boundThrehold=3;  //define the parameter of the Chebyshev's inequality
	double recallThreshold=0; 
	double probablityThreshold=0.95;
	int alpha=3;
	//double boundThreshold2=2.11;// 10%

	// create the output file for the result
	ofstream resultFile(resultFileName, ios::out|ios::app); 
	//resultFile<<"\n------------------------------------------------------------"; 
	allStart=clock();
	if (option==1)
	{
	   cout<<"\n Chebyshev bound"; 
	   if (DisArrayOption==1)
		   cout<<"\n slow version"; 
	   if (DisArrayOption==2)
		   cout<<"\n fast version"; 
	   cout<<"\n MinLength="<<MinK<<", MaxLength="<<MaximalK<<"("<<(double)MaximalK/DIMENSION<<"of "<<DIMENSION<<")"; 
	   cout<<"\n BoundThreshod="<<boundThrehold; 
	   cout<<"\n Alpha="<<alpha; 
	    
	}
	else
	{
	   cout<<"\n KDE bound"; 
	   if (DisArrayOption==1)
		   cout<<"\n slow version"; 
	   if (DisArrayOption==2)
		   cout<<"\n fast version"; 
	   cout<<"\n MinLength="<<MinK<<", MaxLength="<<MaximalK<<"("<<(double)MaximalK/DIMENSION<<"of "<<DIMENSION<<")"; 
	   cout<<"\n ProbablityThreshold="<<probablityThreshold; 
	
	}
		

	for (int cl=0; cl< NofClasses;cl++)
   {  
	   //resultFile<<"\n\n class= "<<cl<<" with label="<<Classes[cl]; 
	   cout<<"\n class="<<cl;
	   int classIndex=cl; // pick the class
	   vector<Feature *> reducedFset; // restore the set of the first round set cover
	   int k=1;
	   clock_t startDisA, endDisA;
	   clock_t startFeature, endFeature;
	   
	   // initialize the DisA and EndP
	   for (int tIndex=ClassIndexes[cl]; tIndex<ClassIndexes[cl]+ClassNumber[cl]; tIndex++)
	   { // start of outer for
		
	         int NumberOfSegments=0; 
	         for (int i=DIMENSION-MinK+1;i>=DIMENSION-MaximalK+1;i--)
			    {
					NumberOfSegments=NumberOfSegments+i;
			     }

			 double ** DisA=new double *[NumberOfSegments]; 
			 int ** EndP= new int *[NumberOfSegments];
			for (int i=0;i<NumberOfSegments;i++)
			{
				DisA[i]=new double[ROWTRAINING];
				EndP[i]=new int[ROWTRAINING]; // ending position array
			}
   
		// Distance array
		string filename="Dis"; 
		string instanceString=IntToString(tIndex) ;
		filename=filename+instanceString; 
		filename.insert(0,path);
		char * f;
		f = new char [filename.size()+1];
		strcpy (f, filename.c_str());

		// Ending position array
		filename="EPc"; 
		filename=filename+instanceString; 
		filename.insert(0,path);
		char * f2;
		f2= new char [filename.size()+1];
		strcpy (f2, filename.c_str());


		ifstream inputFile( f, ios::in);
		ifstream inputFile2( f2, ios::in);
		if ( !inputFile || !inputFile2)
		{
			//cerr << "file does not exist" << endl;
			
			startDisA = clock();
			if (DisArrayOption==1)
			{DisArray(cl, tIndex,  DisA, EndP, NumberOfSegments, ROWTRAINING, MinK, MaximalK);}
			else if (DisArrayOption==2)
			{DisArray2(cl, tIndex,  DisA, EndP, NumberOfSegments, ROWTRAINING, MinK, MaximalK);}
			else
			{DisArray(cl, tIndex,  DisA, EndP, NumberOfSegments, ROWTRAINING, MinK, MaximalK);}

			endDisA=clock();
		//	cout<<"\n\n the disarray is built in "<<(double)(endDisA-startDisA)/CLOCKS_PER_SEC<<" seconds";
		 //   SaveDisArray(DisA, NumberOfSegments, ROWTRAINING, f, EndP, f2);
		}
		else
		{
			loadDisArray(f, DisA, NumberOfSegments, ROWTRAINING);
			loadDisArray(f2, EndP, NumberOfSegments, ROWTRAINING);
		}

		// test  a segment and its threshold
		//Feature * testFeature=ThresholdLearningAll(1676, 3, k, 0,  DisA, 0) ;
		//PrintFeature(testFeature);
		//testFeature=ThresholdLearningFirst(1676, 2.98, k, 0,  DisA, 0) ;
		//PrintFeature(testFeature);
		//testFeature=ThresholdLearningKDE(1676,k,  classIndex, DisA,  0, recallThreshold,  probablityThreshold);
		//PrintFeature(testFeature);

		// compute 
		startFeature=clock();
		totalBmap.reset();   // reset the Bmap for each length's set cover
		vector<Feature *> fSet; 
		//int offset=ClassIndexes[classIndex]*(DIMENSION-k+1)+1;
	    int DisArrayIndex=0;
		for (int currentL=MinK; currentL<=MaximalK;currentL++ )  // for each length
		{   
	      
		   for (int startPosition=0;startPosition<=DIMENSION-currentL;startPosition++)
		  {
		
					Feature * temp;
					if (option==1)
					{  
						temp= ThresholdLearningAll(DisArrayIndex, tIndex, startPosition, boundThrehold, currentL, classIndex, DisA, recallThreshold, EndP, alpha);  
						// temp=ThresholdLearningAll(offset+s, boundThrehold, k, classIndex, DisA, recallThreshold);
					}
					else if (option==2)
					{
					  temp=ThresholdLearningKDE(DisArrayIndex, tIndex, startPosition, currentL, classIndex, DisA, recallThreshold, probablityThreshold,EndP, alpha);
					}
					else
					{
						cout<<"\n invalide option";
						exit(0);
					}

				   if (temp!=NULL)
				   { 
					   fSet.push_back(temp);
				   }
          DisArrayIndex++;
		 }// end of position for
    } // end of length for
   
	//	cout<<"\n the size of feature set of instance="<<tIndex<<" is "<<fSet.size();

		// compute the non-redundent feature set by non-redundant top-K
		while (totalBmap.count()>0)
		{ 
			double max=-1; 
			double maxLength=-1;
			int index=-1; 
			for (unsigned int i=0;i<fSet.size();i++)
			{
				double temp=fSet.at(i)->earlyFScore; 
				if (temp>max )
				{
					max=temp; 
					maxLength=fSet.at(i)->length;
					index=i;
				
				}
			else 
				 if  (temp==max && fSet.at(i)->length>maxLength)
				 {
					max=temp; 
					maxLength=fSet.at(i)->length;
					index=i;
				 }
			}
		   // move this feature to reducedFset 
		   if (index>=0)
		   {
			   Feature * currentFeature=fSet.at(index);
				fSet.erase(fSet.begin()+index); 
				// check if increase the coverage
			
		 
				// check current with the totalBmap if there is 1 in current and 1 in totalBmap, new coverage
				bitset<ROWTRAINING> newCoverage= currentFeature->bmap & totalBmap;

				if (newCoverage.count()>0)
				{
					reducedFset.push_back(currentFeature);
				}
			
				// using bit operation
				for (unsigned int j=0;j<(currentFeature->bmap).size();j++)  // for the current set, update the other set
				{ 
					if ((currentFeature->bmap)[j]==1) 
					{
						totalBmap.reset(j);  // update the total covered set
					}
				}
		   }
		 }// end while  , end of the feature selection in each length
		  cout<<"\n reducedFset of instance:"<<tIndex<<"="<<reducedFset.size();

		  // relase the memory
		 for (int i=0;i<NumberOfSegments;i++)
		 {
			delete [] DisA[i];
		 }
		  delete [] DisA;


		  for (int i=0;i<NumberOfSegments;i++)
		 {
			delete [] EndP[i];
		 }
		  delete [] EndP;
	  
		  for (unsigned int i=0;i< fSet.size();i++)
		  {   
			 //delete [] fSet.at(i)->f; 
			 delete fSet.at(i);
		  }
		  endFeature=clock(); 
	   //   cout<<"\n the features are computed in "<<(double)(endFeature-startFeature)/CLOCKS_PER_SEC<<" seconds";
	 } // end of outer for, end of feature for each isntance

	   //    for (int s=0;s<reducedFset.size();s++)
	   // {
	   // //   PrintFeature(reducedFset.at(s));
		  // PrintFeatureMoreInfo(reducedFset.at(s));
	   //} 


	   // second round of set cover
	   cout<<"\n The total Coverage Rate is"<<(double)allLengthBmap.count()/ClassNumber[classIndex];
	 //  resultFile<<"\n The total Coverage Rate is"<<(double)allLengthBmap.count()/ClassNumber[classIndex];
	   ReduceAllLength(finalFset, reducedFset);
	   cout<<"\n the size of the final set="<<finalFset.size();
	 // resultFile<<"\n the size of the final set="<<finalFset.size();

		for (unsigned int s=0;s<finalFset.size();s++)
		{  
		   cout<<"\n"<<s;
		   //resultFile<<"\n"<<s;
		   PrintFeature(finalFset.at(s));
		   PrintFeature(finalFset.at(s), resultFile); 
		   //PrintFeatureMoreInfo(finalFset.at(s));
	   } 
     
	 //  resultFile<<"\n"; 
	 // classificationAllLength(finalFset, classIndex, resultFile); // CLASSIFY FOR EACH CLASS

	   // UPDATE THE ALL CLASS DATA SET
		while(!finalFset.empty())
	   {  
		  Feature * tempFeature=finalFset.at(0);
          finalFset.erase(finalFset.begin()+0);
		  AllClassesFset.push_back(tempFeature);
	    } 

		
	}// end of for for each classes
    
    for (unsigned int s=0;s<AllClassesFset.size();s++)
		{  cout<<"\n"<<s;
	      // resultFile<<"\n"<<s;
		   PrintFeature(AllClassesFset.at(s));
		   //PrintFeature(AllClassesFset.at(s), resultFile); 

		   //PrintFeatureMoreInfo(finalFset.at(s));
	   } 
    
    //classification
    //classification(reducedFset, classIndex,MaximalK);
 // classificationAllLength(finalFset, classIndex);
	allEnd=clock(); 
	cout<<endl<<(double)(allEnd-allStart)/CLOCKS_PER_SEC<<" finish"<<endl;
	//resultFile<<"\n the total training time is "<<(double)(allEnd-allStart)/CLOCKS_PER_SEC<<" seconds";
	// resultFile<<"\n"; 
   
   clock_t classificationStart=clock();
   classificationAllClasses(AllClassesFset, resultFile);
   clock_t classificationEnd=clock(); 
   cout<<"\n"<<(double)(classificationEnd-classificationStart)/ROWTESTING/CLOCKS_PER_SEC<<" finish2";
   return 0;
}

// second round of feature selection by top-K non-redundant
void ReduceAllLength(vector<Feature *> & finalFset, vector<Feature *> &fSet)
{   
	 while (allLengthBmap.count()>0)
     { 
       double max=-1; 
	   double maxLength=-1;
       int index=-1; 
       for (unsigned int i=0;i<fSet.size();i++)
       { 
		 double temp=fSet.at(i)->earlyFScore; 
         if (temp>max )
         {
			 max=temp; 
			 maxLength=fSet.at(i)->length;
			 index=i;
			
		 }
		 else 
			 if  (temp==max && fSet.at(i)->length>maxLength)
		     {
				max=temp; 
				maxLength=fSet.at(i)->length;
				index=i;
	         }

       }
       // move this feature to reducedFset and reset others 
       if (index>=0)
       {
		   Feature * currentFeature=fSet.at(index);
           fSet.erase(fSet.begin()+index); 
		   // need to remove the redundancy
			
			// check current with the totalBmap if there is 1 in current and 1 in totalBmap, new coverage
			bitset<ROWTRAINING> newCoverage= currentFeature->bmap &allLengthBmap;

			if (newCoverage.count()>0)
			{
				finalFset.push_back(currentFeature);
			}
			
           
        for (unsigned int j=0;j<(currentFeature->bmap).size();j++)  // for the current set, update the other set
        { 
		   if ((currentFeature->bmap)[j]==1) 
           {
              allLengthBmap.reset(j);  // update the total covered set
			 // cout<<"\n reset";
           }
        }

       }
     }// end while
	  cout<<"\n finalFset:"<<finalFset.size();
	   
	  for (unsigned int i=0;i< fSet.size();i++)
	  {   
//	     delete [] fSet.at(i)->f; 
		 delete fSet[i];
	  }
}


void classificationAllClassesWithDefault(vector<Feature *> &Fs, ofstream& resultFile)
{
    int NoClassified=0;
    int CorrectlyClassified=0;
	int sumLength=0;
	//find the default label as the the most frequent class

	int defaultlabel=-1; 
	int mostFrequent=-1;
	for (int ci=0;ci<NofClasses;ci++)
	{
		if(ClassNumber[ci]>mostFrequent)
		{
		  mostFrequent=ClassNumber[ci]; 
		  defaultlabel=Classes[ci];
		}
	}

    for (int i=0;i<ROWTESTING;i++)
    {   
            bool matched=0;
            for (int j=0;j<DIMENSION;j++)// j is the current ending position of the stream
            {    
               for (unsigned int f=0;f<Fs.size();f++)
               { 
                  int tempLength=Fs.at(f)->length;
                  int startingPosition=j-tempLength+1;
                  if (startingPosition>=0)
                  {   
                      double * currentseg=new double[tempLength];
                      for (int ss=0;ss<tempLength;ss++)
                      {currentseg[ss]=testing[i][ss+startingPosition];}
					  double * tempFeatureSeg=getFeatureSegment(Fs.at(f));
                      double tempDis=Euclidean(tempFeatureSeg, currentseg, tempLength);
					  delete [] tempFeatureSeg;
					  delete [] currentseg;
                      if (tempDis<=(Fs.at(f)->threshold))
                      {  //cout<<"\n Instance "<<i<<"("<<labelTesting[i]<<") classified by instance Index="<<Fs.at(f)->instanceIndex<<", position="<<Fs.at(f)->startPosition<<"of length "<<Fs.at(f)->length<< "at ending position "<<j<<" as "<<Fs.at(f)->label;
                         matched=1;
						 predictedLength[i]=j; // store the classified length
						 sumLength=sumLength+j+1;
                         NoClassified++;
                         if(Fs.at(f)->label==labelTesting[i])
                            {CorrectlyClassified++;}
						 break;
                           // break to stop checking more features
                      }
                     
                  }
               }
                if (matched==1) // break the segment loop, finish the current testing example
               { break;}
				
            } // end of for , finish classify the current example by the features
			if(matched==0)  // classified by the default classifier
			{  
				//cout<<"\n Instance "<<i<<"("<<labelTesting[i]<<") classified by default rule  as "<<defaultlabel;
				  if (labelTesting[i]==defaultlabel)
				  {
					  CorrectlyClassified++;
				  }
				  sumLength=sumLength+DIMENSION;
		   }
    }

	//// count the total number in the target class
	//int TargetClassTotal=0;
	//
	//for (int i=0;i<ROWTESTING;i++)
	//{
	//    if (labelTesting[i]==Classes[classIndex])
	//		TargetClassTotal++;

	//}

    /*cout<<"\nthe coverage   is:"<<(double)NoClassified/ROWTESTING; 
    if (NoClassified==0)
    {cout<<"\nthe accuracy  is:"<<0;}
    else*/
    {cout<<"\nthe accuracy  is:"<<(double)CorrectlyClassified/ROWTESTING ;
	cout<<"\n the averaged detection length "<<(double)sumLength/ROWTESTING;}

	/*resultFile<<"\nthe coverage   is:"<<(double)NoClassified/ROWTESTING; 
    if (NoClassified==0)
    {resultFile<<"\nthe accuracy  is:"<<0;}
    else*/
	/*{resultFile<<"\nthe accuracy  is:"<<(double)CorrectlyClassified/ROWTESTING ;
	resultFile<<"\n the averaged detection length "<<(double)sumLength/ROWTESTING;}*/

}
void classificationAllClasses(vector<Feature *> &Fs, ofstream& resultFile)
{
    int NoClassified=0;
    int CorrectlyClassified=0;
	int sumLength=0;
    for (int i=0;i<ROWTESTING;i++)
    {   
            bool matched=0;
            for (int j=0;j<DIMENSION;j++)// j is the current ending position of the stream
            {    
               for (unsigned int f=0;f<Fs.size();f++)
               { 
                  int tempLength=Fs.at(f)->length;
                  int startingPosition=j-tempLength+1;
                  if (startingPosition>=0)
                  {   
                      double * currentseg=new double[tempLength];
                      for (int ss=0;ss<tempLength;ss++)
                      {currentseg[ss]=testing[i][ss+startingPosition];}
					  double * tempFeatureSeg=getFeatureSegment(Fs.at(f));
                      double tempDis=Euclidean(tempFeatureSeg, currentseg, tempLength);
					  delete [] tempFeatureSeg;
					  delete [] currentseg;
                      if (tempDis<=(Fs.at(f)->threshold))
                      { //cout<<"\n Instance "<<i<<"("<<labelTesting[i]<<") classified by instance Index="<<Fs.at(f)->instanceIndex<<", position="<<Fs.at(f)->startPosition<<"of length "<<Fs.at(f)->length<< "at ending position "<<j<<" as "<<Fs.at(f)->label;
						 predictedLabel[i]=(int)Fs.at(f)->label;
                         matched=1;
						 predictedLength[i]=j; // store the classified length
						 sumLength=sumLength+j+1;
                         NoClassified++;
						 cout<<j<<" "<<Fs.at(f)->label<<" "<<endl;
                         if(Fs.at(f)->label==labelTesting[i])
                            {
								CorrectlyClassified++;
						        
						    }
						 break;
                           // break to stop checking more features
                      }
                     
                  }
               }
                if (matched==1) // break the segment loop, finish the current testing example
               { break;}
            }
			if(matched==0)
			cout<<(DIMENSION-1)<<" "<<-4<<" "<<endl;

			
			
    }

	//// count the total number in the target class
	//int TargetClassTotal=0;
	//
	//for (int i=0;i<ROWTESTING;i++)
	//{
	//    if (labelTesting[i]==Classes[classIndex])
	//		TargetClassTotal++;

	//}

    cout<<"\nthe coverage   is:"<<(double)NoClassified/ROWTESTING; 
    if (NoClassified==0)
    {cout<<"\nthe accuracy  is:"<<0;}
    else
    {cout<<"\nthe accuracy  is:"<<(double)CorrectlyClassified/NoClassified; 
	cout<<"\n the averaged detection length "<<(double)sumLength/NoClassified;}

	/*resultFile<<"\nthe coverage   is:"<<(double)NoClassified/ROWTESTING; 
    if (NoClassified==0)
    {resultFile<<"\nthe accuracy  is:"<<0;}
    else
	{resultFile<<"\nthe accuracy  is:"<<(double)CorrectlyClassified/NoClassified; 
	resultFile<<"\n the averaged detection length "<<(double)sumLength/NoClassified;}*/

	// compute the precision and recall of each class;
	for (int c=0;c<NofClasses;c++)
			{     
				  int tp=0; 
				  int classifiedintoC=0; 
				  int hasLabelC=0;
			      for (int i=0;i<ROWTESTING;i++)
				  {
				     if (predictedLabel[i]==Classes[c] && (int)labelTesting[i]==Classes[c])
					 {tp++; }
					 if (predictedLabel[i]==Classes[c] )
					 {classifiedintoC++;}
				     if ((int)labelTesting[i]==Classes[c] )
					 {hasLabelC++;}
				  }

			cout<<"\nClass="<<Classes[c]; 
		    cout<<"\n recall="<<(double)tp/hasLabelC; 
		   cout<<"\n precision="<<(double)tp/classifiedintoC; 
			}
			

}
// classification of one length
void classification(vector<Feature *> &Fs, int classIndex, int k)
{   cout<<"\nk="<<k;
    int TotalCount=0;
    int countDetected=0;
    int countDetectedInTarget=0;
    for (int i=0;i<ROWTESTING;i++)
    {
          bool matched=0;
            for (int j=0;j<=DIMENSION-k;j++)
            {    
                double * currentSegment=new double[k]; 

                for(int jj=0;jj<k;jj++)
                    currentSegment[jj]=testing[i][jj+j];

                for (unsigned int f=0;f<Fs.size();f++)
                { double temp=Euclidean(getFeatureSegment(Fs.at(f)), currentSegment, k);
                  //cout<<"\n instance "<<i<<", the distance is "<<temp;
                   if (temp<=(Fs.at(f)->threshold))
                   {   matched=1;
                       countDetected++;
                       if(Classes[classIndex]==labelTesting[i])
                       {countDetectedInTarget++;}
                       break;}
                }
               if (matched==1)
               { break;}
            }
    }

	// count the total number in the target class
	int TargetClassTotal=0;
	
	
	for (int i=0;i<ROWTESTING;i++)
	{
	    if (labelTesting[i]==Classes[classIndex])
			TargetClassTotal++;

	}

    cout<<"\nthe recall of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/TargetClassTotal; 
    if (countDetected==0)
    {cout<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<0;}
    else
    cout<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/countDetected; 

}


// online classification of all lengths
void classificationAllLength(vector<Feature *> &Fs, int classIndex, ofstream& resultFile)
{
    int TotalCount=0;

    int countDetected=0;
    int countDetectedInTarget=0;
	int sumLength=0;
    for (int i=0;i<ROWTESTING;i++)
    {   
            bool matched=0;
            for (int j=0;j<DIMENSION;j++)// j is the current ending position of the stream
            {    
               for (unsigned int f=0;f<Fs.size();f++)
               { 
                  int tempLength=Fs.at(f)->length;
                  int startingPosition=j-tempLength+1;
                  if (startingPosition>=0)
                  {   
                      double * currentseg=new double[tempLength];
                      for (int ss=0;ss<tempLength;ss++)
                      {currentseg[ss]=testing[i][ss+startingPosition];}
                      double tempDis=Euclidean(getFeatureSegment(Fs.at(f)), currentseg, tempLength);
					  delete [] currentseg;
                      if (tempDis<=(Fs.at(f)->threshold))
                      { //cout<<"\n Instance "<<i<<"("<<labelTesting[i]<<") classified by instance Index="<<Fs.at(f)->instanceIndex<<", position="<<Fs.at(f)->startPosition<<"of length "<<Fs.at(f)->length<< "at ending position "<<j<<" as "<<Fs.at(f)->label;
                         matched=1;
						 predictedLength[i]=j; // store the classified length
						 sumLength=sumLength+j+1;
                         countDetected++;
                         if(Classes[classIndex]==labelTesting[i])
                            {countDetectedInTarget++;}
						 break;
                           // break to stop checking more features
                      }
                     
                  }
               }
                if (matched==1) // break the segment loop, finish the current testing example
               { break;}
            }
    }

	// count the total number in the target class
	int TargetClassTotal=0;
	
	for (int i=0;i<ROWTESTING;i++)
	{
	    if (labelTesting[i]==Classes[classIndex])
			TargetClassTotal++;

	}

    cout<<"\nthe recall of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/TargetClassTotal; 
    if (countDetected==0)
    {cout<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<0;}
    else
    {cout<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/countDetected; 
	cout<<"\n the averaged detection length "<<(double)sumLength/countDetected;}

	/*resultFile<<"\nthe recall of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/TargetClassTotal; 
    if (countDetected==0)
    {resultFile<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<0;}
    else
    {resultFile<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/countDetected; 
	resultFile<<"\n the averaged detection length "<<(double)sumLength/countDetected;}*/

}

// print the information of the feature
void PrintFeature(Feature * f)
{
   if (f!=NULL)
   {
          cout<<"\n the feature=";

    for (int i=0;i<f->length;i++)
	{
		cout<<training[f->instanceIndex][f->startPosition+i]; 
	}
    cout<<"\n instance Index="<<f->instanceIndex;
	cout<<"\n startPosition="<<f->startPosition;
	cout<<"\n length="<<f->length;
    cout<<"\n threshod="<<f->threshold;
    cout<<"\n recall="<<f->recall;
    cout<<"\n precision="<<f->precision;
	cout<<"\n fscore="<<f->fscore;
	
	
    //  cout<<"\n bitmap=";
    // for (int i=0;i<ROWTRAINING;i++)
    //   {cout<<(f->bmap).at(i)<<" ";}
   }
}

// output to the result file
void PrintFeature(Feature * f, ofstream& ResultFile)
{
   if (f!=NULL)
   {
       ResultFile<<"\n the feature=";

    for (int i=0;i<f->length;i++)
	{
		ResultFile<<training[f->instanceIndex][f->startPosition+i]; 
	}
    ResultFile<<"\n instance Index="<<f->instanceIndex;
	ResultFile<<"\n startPosition="<<f->startPosition;
	ResultFile<<"\n length="<<f->length;
    ResultFile<<"\n threshod="<<f->threshold;
    ResultFile<<"\n recall="<<f->recall;
    ResultFile<<"\n precision="<<f->precision;
	ResultFile<<"\n fscore="<<f->fscore;
	ResultFile<<"\n earlyfscore="<<f->earlyFScore;
	// compute the  training example index and the starting position. 
	 int eachLinebase=DIMENSION-f->length+1; 
   
    //  cout<<"\n bitmap=";
    // for (int i=0;i<ROWTRAINING;i++)
    //   {cout<<(f->bmap).at(i)<<" ";}
   }
}
// this function is used to study the precision of best match and all matches
void PrintFeatureMoreInfo(Feature * f)
{
   if (f!=NULL)
   {

    cout<<"\n the feature=";

    for (int i=0;i<f->length;i++)
	{
		cout<<training[f->instanceIndex][f->startPosition+i]; 
	}
    cout<<"\n instance Index="<<f->instanceIndex;
	 cout<<"\n startPosition="<<f->startPosition;
	 cout<<"\n length="<<f->length;
    cout<<"\n threshod="<<f->threshold;
    cout<<"\n recall="<<f->recall<<","<<(f->bmap).count();
    cout<<"\n precision="<<f->precision;
	cout<<"\n fscore="<<f->fscore;
	// compute the precision of all matches
	int countTotal=0; 
	int countTarget=0;
	for (int i=0;i<ROWTRAINING;i++)
		{for (int j=0;j<=DIMENSION-f->length;j++)
		{
		     // compute the Euclidean distance
			double * temp= new double[f->length]; 
			for (int ii=0;ii<f->length;ii++)
			{temp[ii]= training[i][j+ii];}
			double tempDis= Euclidean( getFeatureSegment(f), temp, f->length);
			if (tempDis<=f->threshold)
			{
				  countTotal++; 
				  if (labelTraining[i]==f->label)
				  {
					  countTarget++;
				  }
			}
		}
	} // end of outer for
	cout<<"\n precision of all segments on training="<<(double)countTarget/countTotal;
   } // end of if
} // end of fuction


// print the bitmap
void PrintTotalBitMap()
{
    cout<<"\n the total bitmap=";
    for (int i=0;i<ROWTRAINING;i++)
    {
		 cout<<totalBmap[i]<<" ";
	} 
}

double ComputeFScore(double recall, double precision)
{
	    return 2*recall*precision/ (recall+precision);
}


// learning threshold based on the one tail Chebyshev's inequality
Feature * ThresholdLearningAll(int DisArrayIndex, int instanceIndex, int startPosition,  double m, int k, int targetClass, double ** DisA, double RecallThreshold, int ** EndP, int alpha)  // this index starting from 1, m is the parameter in the bound, k is the length of feature
{
  int classofset=ClassIndexes[targetClass]*(DIMENSION-k+1)+1;
   Feature * currentf= new Feature(); 
   currentf->instanceIndex=instanceIndex; 
   currentf->startPosition=startPosition;
   currentf->length=k;
   currentf->label=Classes[targetClass];
   
   // get the non-target class part in the distance array
   int nonTargetTotal=0; 
   for (int c=0;c<NofClasses;c++)
   {
       if (c!=targetClass)
       {
           nonTargetTotal=nonTargetTotal+ClassNumber[c]; 
       }
   }

   double * nonTargetDis=new double[nonTargetTotal]; 
     
     int i=0;
     for (int c=0;c<NofClasses;c++)
	 {
		   if (c!=targetClass)
		   {   
			   int offset=ClassIndexes[c];
			   for (int e=0;e<ClassNumber[c];e++)
			   {
				   nonTargetDis[i]=DisA[DisArrayIndex][offset+e];
				   i++;
			   }
		   }
	 }
   // compute the mean, standard deviation and the threshold
   double mu=getMean(nonTargetDis, nonTargetTotal); 
   double sd=getSTD(nonTargetDis, nonTargetTotal,mu);
   //cout<<"\nsdNonTarget="<<sd;

   currentf->threshold=mu-m*sd; 
   
   
   delete [] nonTargetDis; // release the memory

   // compute recall, precision and bitmap
   if (currentf->threshold>0)
   {
      int targetCount=0; 
	  double weightedRecall=0;
      int totalCount=0;
      for (int i=0;i<ROWTRAINING;i++)
      { 
		  double temp=DisA[DisArrayIndex][i]; 
         if (temp<=currentf->threshold)
         {
            totalCount++; 
            if (labelTraining[i]==Classes[targetClass])
            {
				targetCount++;
                (currentf->bmap).set(i);  // set the bmap
				weightedRecall=weightedRecall+ pow(((double)1/ EndP[DisArrayIndex][i]),(double)1/alpha);
            }
         }
      }
       currentf->recall=(double)targetCount/ClassNumber[targetClass]; // it is the absolute recall
       currentf->precision=(double)targetCount/totalCount;
	   currentf->fscore=ComputeFScore(currentf->recall, currentf->precision);
	   currentf->earlyFScore=ComputeFScore(weightedRecall, currentf->precision);
      
       if (currentf->recall>=RecallThreshold )
       {   
           for (unsigned int i=0;i<(currentf->bmap).size();i++)
           {
              if ((currentf->bmap)[i]==1)
			  {
				  totalBmap.set(i);  //  set the total Bmap for each length's set cover
			      allLengthBmap.set(i); // set the all length Bmap for the second round of set cover
			  }
           }
           return currentf;
	   }
       else
       {  // delete [] currentf->f; // release the memory
           delete currentf; 
           return NULL;
	   }
   }
   else
   {   
//	   delete [] currentf->f;  // release the memory
       delete currentf;
       return NULL;
   }
   
}

double OneKDE(double * arr, int len, double q, double h, double constant)
{ 
	double temp=0;
	for (int i=0;i<len;i++)
	{
	     temp=temp+exp((arr[i]-q)*(arr[i]-q)* (-1/(2*h*h)));
	}
	
	temp=temp*constant;
	return temp;


}

 //learning the threshold by KDE classification. 
Feature * ThresholdLearningKDE(int DisArrayIndex, int Instanceindex, int startPosition, int k, int targetClass, double ** DisA, double RecallThreshold, double ProbalityThreshold, int ** EndP, int alpha)
{
	 int classofset=ClassIndexes[targetClass]*(DIMENSION-k+1)+1;
	 Feature * currentf= new Feature(); 
	 currentf->instanceIndex=Instanceindex; 
	 currentf->startPosition=startPosition;
	 currentf->length=k;

	 currentf->label=Classes[targetClass];

	 // get the target class part and non-target part in the distance array
   int nonTargetTotal=0; 
   for (int c=0;c<NofClasses;c++)
   {
       if (c!=targetClass)
       {
           nonTargetTotal=nonTargetTotal+ClassNumber[c]; 
       }
   }
   int TargetTotal=ClassNumber[targetClass];

   double * nonTargetDis=new double[nonTargetTotal]; 
   double * TargetDis=new double[TargetTotal];
   double * CurrentDis=new double[ROWTRAINING];

     int nonTargeti=0; 
	 int Targeti=0;
	 int totali=0;
     for (int c=0;c<NofClasses;c++)
	 {    
		   int offset=ClassIndexes[c];
		   if (c!=targetClass)
		   {   
			   for (int e=0;e<ClassNumber[c];e++)
			   {
				   nonTargetDis[nonTargeti]=DisA[DisArrayIndex][offset+e];
				   nonTargeti++;
				   CurrentDis[totali]=DisA[DisArrayIndex][offset+e];
				   totali++;
			   }
		   }
		   else
		   {
		      for (int e=0;e<ClassNumber[c];e++)
			   {
				   TargetDis[Targeti]=DisA[DisArrayIndex][offset+e];
				   Targeti++;
				   CurrentDis[totali]=DisA[DisArrayIndex][offset+e];
				   totali++;
			   }
		   }
	 }
   // compute the mean, standard deviation and the threshold, and optimal h
	//  for the nonTarget Classes
   double muNonTarget=getMean(nonTargetDis, nonTargetTotal); 
   
   double sdNonTarget=getSTD(nonTargetDis, nonTargetTotal,muNonTarget);
   //cout<<"\nsdNonTarget="<<sdNonTarget;
  
   double hNonTarget=1.06* sdNonTarget/ pow (nonTargetTotal,0.2);

   double constantNT=1/(sqrt(2*3.14159265)*nonTargetTotal*hNonTarget);
  //  for the TargetClasses
    double muTarget=getMean(TargetDis, TargetTotal); 
   
   double sdTarget=getSTD(TargetDis, TargetTotal,muTarget);
  
   double hTarget=1.06* sdTarget/ pow (TargetTotal,0.2);
   double constantT=1/(sqrt(2*3.14159265)*TargetTotal*hTarget);

   // sort the totalDis
   quicksort( CurrentDis, 0, ROWTRAINING-1);
  //  compute the Probablity<0; 
   double NegativeTestPoint=-CurrentDis[ROWTRAINING-1]/(ROWTRAINING-1);
   double densityNonTarget=OneKDE(nonTargetDis, nonTargetTotal, NegativeTestPoint, hNonTarget,  constantNT); 
  //cout<<"\n densityNonTarget="<<densityNonTarget;
   double densityTarget=OneKDE(TargetDis, TargetTotal, NegativeTestPoint, hTarget,  constantT);
  // cout<<"\n densityTarget="<<densityTarget;
   double tempTarget=((double)ClassNumber[targetClass]/ROWTRAINING)*densityTarget;
  // cout<<"\n tempTarget="<<tempTarget;
   double tempNonTarget=(1-((double)ClassNumber[targetClass]/ROWTRAINING))*densityNonTarget;
  // cout<<"\n tempNonTarget="<<tempNonTarget;
   double ProTarget=tempTarget/( tempTarget+tempNonTarget);
  // cout<<"\nProbaNegative="<<ProTarget;

   if (ProTarget>ProbalityThreshold)
   {  
	   // compute the breaking Index
       int breakIndex=0;
	   int i=0;
	   for (i=0;i<ROWTRAINING;i++)
	   {  
		  // cout<<"\n"<<i;
		   densityNonTarget=OneKDE(nonTargetDis, nonTargetTotal, CurrentDis[i], hNonTarget,  constantNT); 
		   densityTarget=OneKDE(TargetDis, TargetTotal, CurrentDis[i], hTarget,  constantT); 
		   tempTarget=((double)ClassNumber[targetClass]/ROWTRAINING)*densityTarget; 
		   tempNonTarget= (1-((double)ClassNumber[targetClass]/ROWTRAINING))*densityNonTarget; 
		   ProTarget= tempTarget/( tempTarget+tempNonTarget); 
		   //cout<<"\n"<<i<<" dis="<<CurrentDis[i]<<" Proba="<<ProTarget;
		   if (ProTarget<ProbalityThreshold) // belong to the non-target class
		   {
			  breakIndex=i;
			  break; 
		   }
	   }
	   //if (i==ROWTRAINING)
	   //{currentf->threshold=100000;}
	   //   compute the breaking point between breakingIndex and the previous point
	    if (breakIndex>=1)
	   {
		   int NonofBreakingPoint=20;
		   double value=0;
		   for (value=CurrentDis[breakIndex-1]; value<CurrentDis[breakIndex];value=value+(CurrentDis[breakIndex]-CurrentDis[breakIndex-1])/NonofBreakingPoint)
		   {  
			  // cout<<"\n"<<value;
			   densityNonTarget=OneKDE(nonTargetDis, nonTargetTotal, value, hNonTarget,  constantNT); 
			   densityTarget=OneKDE(TargetDis, TargetTotal, value, hTarget,  constantT); 
			   tempTarget=((double)ClassNumber[targetClass]/ROWTRAINING)*densityTarget; 
			   tempNonTarget= (1-((double)ClassNumber[targetClass]/ROWTRAINING))*densityNonTarget; 
			   ProTarget= tempTarget/( tempTarget+tempNonTarget); 
			   if (ProTarget<ProbalityThreshold) // belong to the non-target class
			   {
				 //  cout<<"\nProba="<<ProTarget;
				  currentf->threshold=value;
				  break; 
			   }
	             
		   }
		   if (value>=CurrentDis[breakIndex])
		   {
			   currentf->threshold=CurrentDis[breakIndex];
		   }
	   }
	   else
	   {
	      currentf->threshold=-1;
	   }
	  
   }
   else
   {
      currentf->threshold=-1;
   }
   
   delete  nonTargetDis; 
   delete TargetDis;
   delete CurrentDis;
 
      if (currentf->threshold>0)
   {
      int targetCount=0; 
	  double weightedRecall=0;
      int totalCount=0;
      for (int i=0;i<ROWTRAINING;i++)
      {  double temp=DisA[DisArrayIndex][i]; 
         if (temp<=currentf->threshold)
         {
            totalCount++; 
            if (labelTraining[i]==Classes[targetClass])
            {
				targetCount++;
                (currentf->bmap).set(i);  // set the bmap
				weightedRecall=weightedRecall+ pow(((double)1/ EndP[DisArrayIndex][i]),(double)1/alpha);
			    
            }
         }
      }
       currentf->recall=(double)targetCount/ClassNumber[targetClass]; // it is the absolute recall
       currentf->precision=(double)targetCount/totalCount;
	   currentf->fscore=ComputeFScore(currentf->recall, currentf->precision);
	   currentf->earlyFScore=ComputeFScore(weightedRecall, currentf->precision);
      
       if (currentf->recall>=RecallThreshold)
       {   
           for (int i=0;i<(currentf->bmap).size();i++)
           {
              if ((currentf->bmap)[i]==1)
			  {
				  totalBmap.set(i);  //  set the total Bmap for each length's set cover
			      allLengthBmap.set(i); // set the all length Bmap for the second round of set cover
			  }
           }
           return currentf;
	   }
       else
       {  // delete [] currentf->f; // release the memory
           delete currentf; 
           return NULL;
	   }
   }
   else
   {   
	  // delete [] currentf->f;  // release the memory
       delete currentf;
       return NULL;
   }
}

// compute the mean
double getMean( double * arr, int len)  // segmentIndex starting from 1
{  
     double sum=0;
     for (int i=0;i<len;i++)
     {
         sum=sum+arr[i];
     }
     return sum/len;
}

// compute the standard deviation given the mean
double getSTD(double * arr, int len, double mean)
{
     double sum=0;
     for (int i=0;i<len;i++)
     {
         sum=sum+(arr[i]-mean)*(arr[i]-mean);
     }
     return sqrt(sum/(len));
}

 // compute the best match distance array of the selected class and the ending position
//  the instanceIndex is the absolute instanceIndex
void DisArray(int classIndex, int instanceIndex, double ** DisArray, int ** EndP, int rows, int columns, int MinK, int MaximalK) 
{   
    // compute the DisArray
	// for (int j=0;j<columns;j++) for each training example
      // cout<<"\n instance ="<<instanceIndex;
	
	   	 for (int tl=0;tl<ROWTRAINING;tl++) // for each training example 
	     {
	   		// listing substring, for each lenght
		    int DisArrayIndex=0;
			
			for (int currentL=MinK; currentL<=MaximalK;currentL++ )  // for each length
			{   
				
				 for (int startPosition=0;startPosition<=DIMENSION-currentL;startPosition++)
				 {  
				    //  cout<<"\ncurrent seg="<<DisArrayIndex;
		                // compute the best match using naive early stopping
					   // compute the best match
					  int bestMatchStartP=-1;
					  double minDis=10000; 
					  for (int l=0;l<DIMENSION-currentL+1;l++)  // a possible match's starting position
					  { 
						 double ret = 0;
						 for (int ii=0; ii<currentL;ii++)
						 {
							  double dist = training[instanceIndex][startPosition+ii]-training[tl][l+ii];
							  ret += dist * dist;
							  if (ret>=minDis)
								  {
									  break; // early stopping
								  }
						  }

						 if (ret<minDis)
						 {
							 minDis=ret;
							 bestMatchStartP=l;
						 }
					  } // end of best match
					 
					  DisArray[DisArrayIndex][tl]=sqrt(minDis); 
					  EndP[DisArrayIndex][tl]=bestMatchStartP+currentL;   // starting from position 1, since 0 will cause infinity
		              DisArrayIndex++;
				 } // end of startPosition for

		}// end of length for
	
	
	}// end of each training example for
} // end of function



// load the best match distance matrix 
void loadDisArray(const char * fileName, double ** Data, int rows, int columns)
{
  ifstream inputFile( fileName, ios::in);
	if ( !inputFile )
	{
		cerr << "file could not be opened" << endl;
		exit(1);
	} // end if
    
	int row=0;
	int col=0;
	while( !inputFile.eof() )
	{
		for ( row=0; row < rows; row++)
			for ( col=0; col < columns; col++)
			{
					inputFile>>Data[row][col];
            }	
	}
	inputFile.close(); 
}

// load the integer array
void loadDisArray(const char * fileName, int ** Data, int rows, int columns)
{
  ifstream inputFile( fileName, ios::in);
	if ( !inputFile )
	{
		cerr << "file could not be opened" << endl;
		exit(1);
	} // end if
    
	int row=0;
	int col=0;
	while( !inputFile.eof() )
	{
		for ( row=0; row < rows; row++)
			for ( col=0; col < columns; col++)
			{
					inputFile>>Data[row][col];
            }	
	}
	inputFile.close(); 
}

// save the best match distance matrix to a text file

void SaveDisArray(double ** DisArray, int rows, int columns, const char * filename, int ** EndP, const char * filename2)
  // save it after compute
{ 
  ofstream outputFile(filename, ios::out);
  ofstream outputFile2(filename2, ios::out);
  for (int row=0; row < rows; row++)
  {
     for (int col=0; col < columns; col++)
    {
       outputFile<<DisArray[row][col];
       outputFile<<" ";

	   outputFile2<<EndP[row][col];
       outputFile2<<" ";
     }
       outputFile<<endl;
	   outputFile2<<endl;
  }   
    outputFile.close();
	outputFile2.close();
}





// print training data
void printData(int index)
{
  cout<<"\nlabel="<<labelTraining[index]<<" "; 
  for (int i=0;i<DIMENSION;i++)
  {cout  <<training[index][i]<<" ";}
}

// load the training and the testing data from text file
void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[],int len  )
{
    
	ifstream inputFile( fileName, ios::in);
	if ( !inputFile )
	{
		cerr << "file could not be opened" << endl;
		exit(1);
	} // end if

	int row=0;
	int col=0;
	while( !inputFile.eof() )
	{
		for ( row=0; row < len; row++)
			for ( col=0; col < DIMENSION+1; col++)
			{
				if (col==0)
				{  
					inputFile>>Labels[row];
				}
				else
				{
					inputFile>>Data[row][col-1];
				}
			}
	}

	inputFile.close();
}

// the function of int to string
string IntToString(int intValue) 
{
	  string myBuff;
	  string strRetVal;
	  // Create a new char array
	  myBuff = new char[100];
	  // Set it to empty
	  //memset(myBuff,'\0',100);
	  // Convert to string
	  myBuff = std::to_string(intValue);
	  //itoa(intValue,myBuff,10);
	  // Copy the buffer into the string object
	  strRetVal = myBuff;
	  // Delete the buffer
	  return(strRetVal);
}


double * getFeatureSegment(Feature *f)
{
	double * temp=new double[f->length]; 
	for (int i=0;i<f->length;i++)
	{
		temp[i]=training[f->instanceIndex][f->startPosition+i]; 
	}

	return temp;

}

//the following code compute the DisArray
// compute original matrix
void createOriginalMatrix(double * array1, int iLengthOfArray1,
                          double * array2, int iLengthOfArray2,
                          double * &pMatrix)
{
    if(pMatrix!=NULL || array1==NULL || array2==NULL)
        return;

    pMatrix = new double[iLengthOfArray1*iLengthOfArray2];

    //using Array1 for horizontal row, Array2 in vertical
    
    for(int j=0; j<iLengthOfArray2; j++)
    {
        for(int i=0; i<iLengthOfArray1; i++)
        {        
            pMatrix[j*iLengthOfArray1+i] = (array1[i]-array2[j])*(array1[i]-array2[j]);
        }
    }

   /* cout<<"createOriginalMatrix: "<<endl;    
    for(int j=0; j<iLengthOfArray2; j++)
    {
        for(int i=0; i<iLengthOfArray1; i++)
        {
            cout<<pMatrix[i + j*iLengthOfArray1]<<";\t";
        }
        cout<<endl;
    }*/
}

void computeWholeSubstringArray(double * array1, int iLengthOfArray1,
                                double* array2, int iLengthOfArray2,
                                double * pMatrixOriginal,
                                double * pArraySubstring, int * pArraySubstringEndingPosition, int iLengthOfSubstringArray, int MaximalLength)
{
    if(pArraySubstring==NULL || array1==NULL || array2==NULL)
        return;

    double* tempMatrix = new double[iLengthOfArray1*iLengthOfArray2];
    int iIndexofSubstring=0;
    int i, j, k;
    //clear out the substring score array
    for(i = 0; i<iLengthOfArray1*iLengthOfArray2; i++)
    {
        tempMatrix[i] = 0;
    }

    //from the length of 1 ot length of Array1
    for(i=0; i<MaximalLength; i++)  // this is the different length of segment
    {
        //update the tempMatrix
        for(j=0;j<iLengthOfArray1-i; j++)  // the start position of array1
        {
            double fMinimum=0;
            int iMinimumEndingPosition =0;
            for(k=0;k<iLengthOfArray2-i; k++)
            {

                double fTempValue= tempMatrix[k*iLengthOfArray1 + j] + pMatrixOriginal[(k+i)*iLengthOfArray1 + (j+i)];
                if( k == 0 )
                {
                    fMinimum = fTempValue;
                    iMinimumEndingPosition = k+i+1;
                }
                else
                {
                    if(fTempValue<fMinimum)
                    {
                        fMinimum = fTempValue;
                        iMinimumEndingPosition = k+i+1;
                    }
                }
                tempMatrix[k*iLengthOfArray1 + j]=fTempValue;
            }
            pArraySubstring[iIndexofSubstring] = sqrt(fMinimum);
            pArraySubstringEndingPosition[iIndexofSubstring] = iMinimumEndingPosition;
            iIndexofSubstring++;
        }

        /*cout<<"Temp matrix for: length "<<i+1<<endl;
        for(k=0;k<iLengthOfArray2-i; k++)
        {
            for(j=0;j<iLengthOfArray1-i; j++)
            {
                cout<<tempMatrix[k*iLengthOfArray1 + j]<<";\t";
            }
            cout<<endl;
        }*/

    }
    
    /*int index = 0;
    for(int m=0; m<iLengthOfArray1; m++)
    {
        cout<<"result for: length "<<m+1<<endl;
        for(int n=0;n<iLengthOfArray1-m; n++)
        {
            cout<<pArraySubstring[index]<<":"<<pArraySubstringEndingPosition[index]<<";\t";
            index++;
        }
        cout<<endl;
    }*/
	delete [] tempMatrix;
}
	// compute the best match distance array of the selected class and the ending position
//  the instanceIndex is the absolute instanceIndex
void DisArray2(int classIndex, int instanceIndex, double ** DisArray, int ** EndP, int rows, int columns, int MinK, int MaximalK) 
{   
    // compute the DisArray
	// for (int j=0;j<columns;j++) for each training example
      // cout<<"\n instance ="<<instanceIndex;
	
    for (int tl=0;tl<ROWTRAINING;tl++) // for each training example 
	{
	  double * pArraySubstring=NULL;  // the distances
      int * pArraySubstringEndingPosition = NULL;  // th ending position
      int iArraySubstringArrayLength = 0;
      int iArraySubstringArrayEndingPositionLength = 0;	

	  double * pMatrixOriginal = NULL;
      createOriginalMatrix(training[instanceIndex],DIMENSION, training[tl],DIMENSION, pMatrixOriginal);
	  int pLengthOfSubstringArray =0; 
	  for (int c=DIMENSION;c>=DIMENSION-MaximalK+1;c--)
	  {
		  pLengthOfSubstringArray=pLengthOfSubstringArray+c;
	  }
      pArraySubstring = new double[pLengthOfSubstringArray];
      pArraySubstringEndingPosition = new int[pLengthOfSubstringArray];
      computeWholeSubstringArray(training[instanceIndex], DIMENSION, training[tl], DIMENSION, pMatrixOriginal, pArraySubstring, pArraySubstringEndingPosition, iArraySubstringArrayLength, MaximalK);
      
	  // how many from 1 to MinK
	  int offset=0; 
	  for (int c=DIMENSION; c>=DIMENSION-(MinK-1)+1;c--)
	  {offset=offset+c;}
	  
	  // copy into the disarray and the pend array
	  for(int i=offset;i<pLengthOfSubstringArray;i++)
	  {
	      DisArray[i-offset][tl]=pArraySubstring[i];
		  EndP[i-offset][tl]=pArraySubstringEndingPosition[i];
	  }
	  
	// release the memory
	delete[] pMatrixOriginal;
    pMatrixOriginal = NULL;


    delete[] pArraySubstring;
    delete[] pArraySubstringEndingPosition;
    pArraySubstringEndingPosition = NULL;
    pArraySubstring = NULL;
     pMatrixOriginal = NULL;
		
		
	
	
	}// end of each training example for
} // end of function

