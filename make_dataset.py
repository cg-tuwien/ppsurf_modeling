import framework_main

if __name__ == '__main__':

    framework_params = [
        '--schema', 'source/framework/configSchema.xsd',
        '--xml', 'source/framework/p2s_abc_modeling_course.xml',
        '--log_dir', 'logs_framework',
        # '--worker_processes', '4',
        '--worker_processes', '0',
    ]

    args = framework_main.parse_arguments(framework_params)
    framework_main.framework_main(args)
    print('done!')
