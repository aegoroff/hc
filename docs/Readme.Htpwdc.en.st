$appName$ Readme 

$appName$ is a simple console application to crack Apache passwords files (.htpasswd) 
using brute force method. Some application's features:

- Crack all hashes from file specified
- Crack one hash specified using corresponding parameter
- List .htpasswd file accounts, so as to get the login to crack
- Crack hash passed through commind line (-h or --hash parameter)

Also there are:

- определить словарь для подбора (по умолчанию это все цифры и буквы в верхнем и нижнем регистрах)
- определить минимальную длину пароля (по умолчанию 1)
- определить максимальную длину пароля (по умолчанию 10)

Usage

$exeName$ [OPTION] ...

Available options:

  -a [ --dict ] <строка>        словарь пароля. По умолчанию это все цифры и буквы в верхнем и нижнем регистрах
                                в верхнем и нижнем регистрах
  -n [ --min ] <число>          The minimal length of the password to crack. 1 by default
  -x [ --max ] <число>          The maximal length of the password to crack. 10 by default
  -f [ --file ] <путь>          full path to password's file
  -h [ --hash ] <hash>          hash to crack
  -l [ --login ] <логин>        login from password file to crack password for
  -i [ --list ]                 list accounts from .htpasswd file
  -? [ --help ]                 Show help
  
Examples