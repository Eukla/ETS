void quicksort( double a[], int start, int end)
{
	// the base case

     if (start>=end)
		 return;
	 double pivot=a[end]; // use the last element as the pivot; 
     
	 int storedIndex=start; // the first element of S2
	 for (int i=start;i<end;i++)
	 {
	    if (a[i]<pivot)
		{
		  
			
			    // swap with stored Index
				double temp=a[storedIndex]; 
				a[storedIndex]=a[i]; 
				a[i]=temp; 
				storedIndex++; 
			
		
		}
	 }
	  // swap a[storedIndex] with pivot
		double temp=a[storedIndex]; 
		a[storedIndex]=a[end]; 
		a[end]=temp; 
		/*cout<<"\n";
	for (int i=0;i<10;i++)
	{
	   cout<<a[i]<<" ";
	}*/
	// recursive call 
		quicksort( a, start, storedIndex-1); 
		quicksort(a, storedIndex+1, end); 
	 
}
