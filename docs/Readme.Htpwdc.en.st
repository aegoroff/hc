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
                                letters both low and upper case by default.It's possible 
                                using symbol classes short form. 0-9 all digits, a-z low case latin letters, 
                                A-Z upper case latin letters. Symbol classes order doesn't matter.
  -n [ --min ] <number>         The minimal length of the password to crack. 1 by default
  -x [ --max ] <number>         The maximal length of the password to crack. 10 by default
  -t [ --threads ] <number>     Set the number of working threads.
                                The half of system processors by default.
  -f [ --file ] <path>          full path to password's file
  -h [ --hash ] <hash>          hash to crack
  -l [ --login ] <login>        login from password file to crack password for
  -i [ --list ]                 list accounts from .htpasswd file
  -? [ --help ]                 Show help
  
Examples

Crack all logins from file specified

$exeName$.exe -f file.txt


Crack only login "login1"

$exeName$.exe -f file.txt -l login1


Listing all logins from file

$exeName$.exe -f file.txt -i


Crack all logins from file specified using custom dictionary, min and max password length

$exeName$.exe -f file.txt -a 0123456789 -n 2 -x 5


Crack all logins from file specified using custom dictionary in short form (digits and low case latin letters)

$exeName$.exe -f file.txt -a 0-9a-z


Crack hash passed through command line

$exeName$.exe -h {SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=