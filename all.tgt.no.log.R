library(combinat)
library(expm)


#number of individuals#
nrow.mat <- sum(mat[,1])

#split.mat function helps split colms of the matrix into vectors in list.vec#
split.mat <- function(mat){
  list.vec <-list()
  for (i in 1:ncol(mat)){
    list.vec[[i]] <- mat[,i]
  }
  return (list.vec)
}

out.list <- split.mat(mat=mat)
#converts into one possible complete dataset according to each count#
one.path <- function(mat.in.list){
  list.panel <-list()
  dummy.fun <-function(j){
    rep(j,mat.in.list[[i]][j])
  }
  for (i in 1:length(mat.in.list)){
    list.panel[[i]] <- sapply(c(1:length(mat.in.list[[1]])),dummy.fun)
  }
  
  return (list.panel)
}

one.c.s <- matrix(unlist(one.path(out.list)),byrow=FALSE,nrow=nrow.mat)

#we want split one.c.s matrix by colm, s.t. we can permute it's outter colms#
try3 <- function(mat){
  dummy3 <- list()
  for (i in 1:ncol(mat)){
    dummy3[[i]] <- mat[,i]
  }
  return (dummy3)
}

#one.c.s splitted by colms#
interm <- try3(one.c.s)

#maze function gives us all permutations of colms without the 1's#
maze <- function (i,mat=one.c.s,mat.vec=interm){
  perm.route <- list()
  
  if(length(mat.vec[[ncol(mat)-i]][which(mat.vec[[ncol(mat)-i]] != 1)])<=1){
    perm.route <- mat.vec[[ncol(mat)-i]][which(mat.vec[[ncol(mat)-i]] != 1)]
    return (perm.route)
  }
  else{
    perm.route <- unique(permn(mat.vec[[ncol(mat)-i]][which(mat.vec[[ncol(mat)-i]] != 1)]))
    return(perm.route)
  }  
}

all.colm.perm <- lapply(c(0:(ncol(mat)-2)),maze)
#note Var1 is the outter most perm, and the ... last Var is the second inner most colm#
#note the length of this data frame is the number of Vars#
#note each colm/row is still a list format#
all.comb.permed <- expand.grid(all.colm.perm)

#now we need to input all permute path into the matrix to form each complete dataset#
one.mat <- function(x,mat=one.c.s,mat.vec=interm,perm.route=all.comb.permed){
  for (i in 0:(ncol(mat)-2)){
    mat.vec[[ncol(mat)-i]][which(mat.vec[[ncol(mat)-i]] != 1)] <-  unlist(perm.route[x,][[i+1]])
  }
  return (mat.vec)
}

#this function put generates all permute pathed matrix into a complete list#
out.all.mat <- function(x){
  mat.final <- list()
  
  for (i in 1:nrow(all.comb.permed)){
    mat.final[[i]] <- matrix(unlist(one.mat(x=i)),byrow=FALSE,nrow=nrow(one.c.s)) 
  } 
  return (mat.final)
}

each.mat <- out.all.mat()

#now we want to permute all rows within each matrix, and delete all duplicates#

#this function gets each vector of the one.c.s matrix out, then do row permute#
try2 <- function(mat){
  dummy2 <- list()
  for (i in 1:nrow(mat)){
    dummy2[[i]] <- mat[i,]
  }
  return (dummy2)
}

#row permute#
#gives all complete paths in list#

tryout <- function(each.mat = each.mat){
  dummy3<-list()
  com.data.set <- function(x){
    intermed <- unique(permn(try2(each.mat[[x]])))
  }
  list.all <- sapply(c(1:length(each.mat)),com.data.set)
  
  dummy4 <- list()
  
  if (nrow.mat >5){
    for (j in 1:length(list.all)){
      for(z in 1:length(list.all[[j]])){
        dummy4[[z]]<-matrix( unlist(list.all[[j]][[z]]) , byrow=T, nrow=nrow.mat)   
      }
      dummy3 <- append(dummy3,dummy4)
    }
  }
  
  else{
    for (j in 1:length(list.all)){
      dummy3[[j]]<-matrix( unlist(list.all[[j]]) , byrow=T, nrow=nrow.mat)   
    }
  }
  
  return(dummy3)
}

#delets all duplicates#
#all.path<- unique(tryout(each.mat))
system.time(all.path<- unique(tryout(each.mat)))


p.for.q <- function(lamda,t=1){
  q<- matrix(c(-lamda[1],lamda[1],0,
               0,-lamda[2],lamda[2],
               0,lamda[3],-lamda[3]),nrow=3,byrow=T)
  p <- expm(q)
}

#the purpose of this function is to construct a matrix of how many of each Pij's we have in a given mat#
#states = how many states are there in you mat#
count.p <- function(mat,states){
  #transpose of the mat give you row wise diff#
  mat <-t(mat)
  mat.diff <- diff(mat)
  mylist <- list()
  
  #take all values that's not 1 out#
  for(i in 1:ncol(mat.diff)){
    if(length(which(mat.diff[,i]!=0)) == 0) {
      mylist[i] <-NULL
    }
    else{
      mylist[[i]] <- mat.diff[,i][which(mat.diff[,i]!=0)[1]:nrow(mat.diff)]
    }
  }
  #this gives you how many 1->1's you may have #
  number_at_1 <- length(mat.diff)-length(unlist(mylist))
  
  #here we construct a empty matrix#
  mat.count <- matrix(rep(0,states^2),nrow=states)
  
  #input number_at_1 into the [1,1]th#
  #mat.count[1,1] <- number_at_1 
  
  #now we find the rest 1->2, 1->3's etc...#
  prob.list <- list()
  path.count <- function(x){
    dummy <- cumsum(mylist[[x]])
    #we want to add 0 to the head of each list[[x]]#
    vec.0 <- rep(0,(length(dummy)+1))
    vec.0[1]<-1
    #now we have the original dataset back#
    vec.0[which(vec.0==0)]<-dummy+1
    
    #gets all Pij counts for individual i#
    for (i in 2:length(vec.0)){
      #each time, a certain place is encountered, we add one to include that individual#
      mat.count[vec.0[i-1],vec.0[i]] <- mat.count[vec.0[i-1],vec.0[i]] + 1
    }
    #return product of Pij for individual i#
    return (mat.count)    
  } 
  indiv.count <- lapply(c(1:length(mylist)),path.count)
  mat.count <- Reduce('+',indiv.count)
  mat.count[1,1] <- number_at_1
  #remove all NA's#
  #NA is present because some paths never deviate from state 1#
  #note need to include p11's probs#
  #result <- indiv.[!is.na(indiv.prod)]
  
  return(mat.count)  
}

#mat here indicates indivial path matrix, i.e. one complete dataset#
Weights <- function(mat,lamda,states){
  
  #this p gives all the numercial values of p(t) with the given lamda#
  p <- p.for.q(lamda=lamda)
  
  #mat.count gives how many of each pij's are indeed in a given mat#
  mat.count <- count.p(mat=mat,states=states)
  
  #since state 1 is transient,no other states can be before1#
  #we assume that each indiv is indep in this case#
  #let p11=P(state at time r =1 | state at time r-1 = 1)#
  #time-homo#
  
  #make pij's a list by row, also the mat.count a list by row, then the mat.count serves as a vector of power#
  mat.power <- as.vector(mat.count)
  p.vec <- as.vector(p)
  
  dummy<-p.vec^mat.power
  result <- prod(dummy[which(dummy != 0)])
  
  return(result)  
}

e.step <- function(all.path,states,lamda){
  #prod.prob gives all weights for all complete data sets#
  prod.prob <- c()
  for (i in 1:length(all.path)){
    prod.prob[i] <- Weights(mat=all.path[[i]],states=states,lamda=lamda)
  }
  #we want the fraction of contribution of each#
  total <- sum(prod.prob)
  frac.prod.prob <-c()
  frac <- function(x){
    frac.prod.prob[x] <- prod.prob[x]/total
  }
  ws <-sapply(c(1:length(prod.prob)),frac)
  return (ws)
}

m.step <- function(lamda_init,ws){
  object <- function(lamda,x){
    p<- p.for.q(lamda)
    
    mat.count <- count.p(mat=all.path[[x]],states=3)
    
    #make pij's a list by row, also the mat.count a list by row, then the mat.count serves as a vector of sum#
    mat.sum <- as.vector(mat.count)
    p.vec <- as.vector(p)
    dummy <- p.vec*mat.sum
    ws[x]*sum(log(dummy[which(dummy != 0)]))*(-1)  
  }
  my.array<-list()
  
  ah <- function(lamda){
    for (i in 1:length(all.path)){
      mat.count <- count.p(mat=all.path[[i]],states=3)  
      my.array[i] <- object(lamda,x=i)
      
    }
    final.fun <- Reduce('+',my.array)
  }
  #lamda_new <- optim(par = lamda_init, fn = ah, lower=c(1e-5,1e-5,1e-5))
  lamda_new <- nlminb(lamda_init, ah, lower=c(1e-5,1e-5,1e-5))
  
  return (lamda_new$par)
}

#before running em:
############run expand.grid to obtain all.path###########
############run count.p to obtain all path counts###########
############run Weights and e.step ###########
############run p.for.q to obtain q matrix###########
############run m.step ###########

em <- function(agg.mat, tol = 0.05, max.step = 1e2){
  step <- 0
  
  #how many states are there in the matrix#
  states <- nrow(agg.mat)
  
  #all possible paths from the given agg.mat#
  #run expand.grid to obtain all.path#
  
  #initial lambdas#
  lamda_init <- c(5,5,5)
  
  repeat{
    
    #E-step#    
    ws <- e.step(all.path,states=states,lamda=lamda_init)
      
    old.lamdas <- lamda_init
    
    lamda_init <- m.step(old.lamdas,ws=ws)
    
    #convergence achieved
    if (norm(as.matrix(old.lamdas-lamda_init), type="F") < tol) {break} 

    step = step + 1
    if (step > max.step) {break}
     
  }
  return(lamda_init)
  
}

#realize the lamda1 is always the same, however, the other two is not?#

system.time(output <- em(agg.mat = mat))


