$appName$ Readme 

$appName$ is a simple console application to crack Apache passwords files (.htpasswd) 
using brute force method. Some application's features:

- Crack all hashes from file specified
- Crack one hash specified using corresponding parameter
- List .htpasswd file accounts, so as to get the login to crack
- Crack hash passed through commind line (-h or --hash parameter)

Also there are:

- cracking dictionary defining (all digits and latin letters both low and upper case by default)
- minimal password's length setting (1 by default)
- maximum password's length setting (10 by default)

Usage

$exeName$ [OPTION] ...

Available options:

  -a [ --dict ] <string>        cracking dictionary. All digits and latin 
                                letters both low and upper case by default
  -n [ --min ] <number>         The minimal length of the password to crack. 1 by default
  -x [ --max ] <number>         The maximal length of the password to crack. 10 by default
  -f [ --file ] <path>          full path to password's file
  -h [ --hash ] <hash>          hash to crack
  -l [ --login ] <login>        login from password file to crack password for
  -i [ --list ]                 list accounts from .htpasswd file
  -? [ --help ]                 Show help
  
Examples