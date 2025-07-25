import pytest
from unittest.mock import patch, MagicMock
from fp_orchestrator_utils.cli.main import main

class TestCLIMain:
    """ Test CLI main module. """

    def test_main_with_no_args(self, capsys):
        """ Test main function with no arguments. """
        with patch('sys.argv', ['fp_orchestrator_utils']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2 # Argument parsing error
            captured = capsys.readouterr()
            assert "arguments are required: command" in captured.err

    def test_main_with_help_flag(self, capsys):
        """ Test main function with help flag. """
        with patch('sys.argv', ['fp_orchestrator_utils', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0 # Help should exit with code 0
            captured = capsys.readouterr()
            assert "usage: fp_orchestrator_utils" in captured.out
            assert "FP Proto Manager CLI Tool" in captured.out

    @patch('fp_orchestrator_utils.cli.proto_commands.ProtoManager')
    def test_main_proto_download_success(self, mock_proto_manager):
        """ Test main function with proto download command. """
        mock_manager = MagicMock()
        mock_manager.check_connection.return_value = True
        mock_proto_manager.return_value = mock_manager

        with patch('sys.argv', ['fp_orchestrator_utils', 'proto', 'download']):
            result = main()
            assert result == 0

            mock_manager.check_connection.assert_called_once()
            mock_manager.download_protos.assert_called_once()

    @patch('fp_orchestrator_utils.cli.proto_commands.ProtoManager')
    def test_main_proto_upload_success(self, mock_proto_manager):
        """ Test main function with proto upload command. """
        mock_manager = MagicMock()
        mock_manager.check_connection.return_value = True
        mock_proto_manager.return_value = mock_manager

        with patch('sys.argv', ['fp_orchestrator_utils', 'proto', 'upload']):
            result = main()
            assert result == 0

            mock_manager.check_connection.assert_called_once()
            mock_manager.upload_protos.assert_called_once()

    @patch('fp_orchestrator_utils.cli.proto_commands.ProtoManager')
    def test_main_proto_generate_success(self, mock_proto_manager):
        """ Test main function with proto generate command. """
        mock_manager = MagicMock()
        mock_manager.check_connection.return_value = True
        mock_proto_manager.return_value = mock_manager

        with patch('sys.argv', ['fp_orchestrator_utils', 'proto', 'generate']):
            result = main()
            assert result == 0

            mock_manager.generate_grpc_code.assert_called_once()
    
    @patch('fp_orchestrator_utils.cli.proto_commands.ProtoManager')
    def test_main_proto_check_success(self, mock_proto_manager):
        """ Test main function with proto check command. """
        mock_manager = MagicMock()
        mock_manager.check_connection.return_value = True
        mock_proto_manager.return_value = mock_manager

        with patch('sys.argv', ['fp_orchestrator_utils', 'proto', 'check']):
            result = main()
            assert result == 0

            mock_manager.check_connection.assert_called_once()
    
    def test_main_invalid_command(self, capsys):
        """ Test main function with an invalid command. """
        with patch('sys.argv', ['fp_orchestrator_utils', 'invalid_command']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
            captured = capsys.readouterr()
            assert "invalid_command" in captured.err

    def test_main_proto_without_subcommand(self, capsys):
        """ Test main function with proto command but no subcommand. """
        with patch('sys.argv', ['fp_orchestrator_utils', 'proto']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
            captured = capsys.readouterr()
            assert "the following arguments are required: proto_command" in captured.err

    @patch('fp_orchestrator_utils.cli.proto_commands.ProtoManager')
    def test_main_with_exception_handling(self, mock_proto_manager):
        """ Test main function with exception handling in commands. """
        mock_manager = MagicMock()
        mock_manager.check_connection.side_effect = Exception("Connection error")
        mock_proto_manager.return_value = mock_manager

        with patch('sys.argv', ['fp_orchestrator_utils', 'proto', 'download']):
            result = main()
            assert result == 1

            mock_manager.check_connection.assert_called_once()
            mock_manager.download_protos.assert_not_called()